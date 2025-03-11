import os
import librosa
import numpy as np
import logging
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch
from pydub import AudioSegment
import io
import json
from transformers import TrainerCallback
import warnings


# Update PATH to include /opt/homebrew/bin
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
# print("Updated PATH:", os.environ["PATH"])

# Set ffmpeg and ffprobe paths for pydub
AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"

warnings.filterwarnings("ignore")

def prepare_dataset(audio_dir, transcript_dir):
    audio_files = []
    transcripts = []

    print(f"Scanning directory: {audio_dir}")
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".mp3"):
            audio_file = os.path.join(audio_dir, file_name)
            transcript_file = os.path.join(transcript_dir, file_name.replace(".mp3", ".txt"))

            # print(f"\nProcessing file: {file_name}")
            # print(f"Audio path: {audio_file}")
            # print(f"Transcript path: {transcript_file}")

            if not os.path.exists(audio_file):
                print(f"Error: Audio file does not exist: {audio_file}")
                continue

            if not os.path.exists(transcript_file):
                print(f"Error: Transcript file does not exist: {transcript_file}")
                continue

            try:
                # Load audio with pydub
                audio_segment = AudioSegment.from_mp3(audio_file)
                audio_array = np.array(audio_segment.get_array_of_samples())
                audio = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                sr = audio_segment.frame_rate
                # print(f"Loaded audio shape: {audio.shape}, Sample rate: {sr}")

                if len(audio.shape) == 0 or audio.size == 0:
                    print("Error: Audio data is empty after loading")
                    continue

                # Load and check transcript
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()

                if not transcript:
                    print(f"Error: Transcript is empty: {transcript_file}")
                    continue

                # print("Successfully loaded audio and transcript")
                audio_files.append(audio_file)
                transcripts.append(transcript)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                import traceback
                print(traceback.format_exc())

    print(f"\nTotal valid pairs found: {len(audio_files)}")
    return {"audio": audio_files, "transcript": transcripts}

def preprocess_function(examples, processor):
    min_length_samples = 3000 * 160  # 3000 frames at 16kHz, each frame ~160 samples
    audio_arrays = []

    for audio_path in examples["audio"]:
        audio_segment = AudioSegment.from_mp3(audio_path)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        audio, _ = librosa.load(io.BytesIO(wav_io.read()), sr=16000)

        # Check and pad to ensure minimum length
        # Whisper expects the mel input features to be of length 3000, but found 2408
        if len(audio) < min_length_samples:
            padding = min_length_samples - len(audio)
            audio = np.pad(audio, (0, padding), "constant")

        audio_arrays.append(audio)
    
    # Process audio (the processor will handle truncation to 3000 if necessary)
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    # print(f"Processed input shape: {inputs.input_features.shape}")

    labels = processor.tokenizer(
        examples["transcript"],
        padding="longest",  # Dynamically pad
        truncation=True,    # Enable truncation
        max_length=448,     # Set the max length explicitly
        return_tensors="pt",
    ).input_ids

    labels[labels == processor.tokenizer.pad_token_id] = -100  # Replace pad tokens



    # Replace padding token ID

    # The Whisper model’s training loss function (e.g., CrossEntropyLoss) uses the labels to compute how well the model's predicted outputs match the ground truth. By replacing padding tokens with -100:
    # The loss function will ignore positions marked as -100.
    # This prevents the model from being penalized for predictions at padding positions, which are meaningless.
    # Critical step to ensure that the padding tokens in the transcript labels are ignored during the loss computation in training.

    labels = labels.numpy() # Convert PyTorch tensor to NumPy array
    labels[labels == processor.tokenizer.pad_token_id] = -100 # Replace <pad> token IDs with -100

    # calculating the loss on both the training data (for optimization) 

    return {
        "input_features": inputs.input_features.numpy(),
        "labels": labels,
    }



class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = torch.tensor([feature["input_features"] for feature in features]).contiguous()
        label_features = [torch.tensor(feature["labels"]).contiguous() for feature in features]

        label_features = torch.nn.utils.rnn.pad_sequence(
            label_features, batch_first=True, padding_value=-100
        ).contiguous()

        # Debugging shapes
        print(f"Input features shape: {input_features.shape}")
        print(f"Label features shape: {label_features.shape}")

        return {
            "input_features": input_features,
            "labels": label_features,
        }
    
class LoggerCallback(TrainerCallback):
    def __init__(self, log_file="training_logs.json"):
        self.log_file = log_file

        # Initialize the log file
        with open(self.log_file, "w") as f:
            json.dump([], f)


    #  CHANGES HERE
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Filter out unnecessary keys
            log_entry = {
                "step": state.global_step,
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
                "epoch": logs.get("epoch"),
            }

            if "eval_loss" in logs:
                log_entry["val_loss"] = logs.get("eval_loss")
            if "eval_wer" in logs:  # Example metric
                log_entry["val_wer"] = logs.get("eval_wer")
            
            # Append log to the file
            with open(self.log_file, "r+") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=4)

            # Optionally print the log
            # print(f"Step {state.global_step}: {log_entry}")

def main():
    # Paths to dataset
    audio_path = "/Users/hassanshuman/SaraIRPLocal/FINAL MP3"
    transcripts_path = "/Users/hassanshuman/SaraIRPLocal/FINAL TXT"

    # Load model and processor
    print("Loading model and processor...")
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Prepare the dataset
    print("Preparing dataset...")
    data_dict = prepare_dataset(audio_path, transcripts_path)
    print(f"Found {len(data_dict['audio'])} valid audio-transcript pairs")

    # Create Dataset
    dataset = Dataset.from_dict(data_dict)

    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Test Dataset: {len(test_dataset)} samples")

    # Process dataset
    print("Processing dataset...")

    # Process train dataset
    encoded_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, processor),
        remove_columns=dataset.column_names,
        batch_size=8,
        batched=True,
    )

    # Process test dataset
    encoded_test_dataset = test_dataset.map(
        lambda examples: preprocess_function(examples, processor),
        remove_columns=dataset.column_names,
        batch_size=8,
        batched=True,
    )

    data_collator = WhisperDataCollator(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-finetuned",
        per_device_train_batch_size=4,  # Increase if GPU memory allows; otherwise keep it at 1
        learning_rate=3e-5,  # Slightly higher learning rate for fine-tuning specific domain
        num_train_epochs=20,  # Increase to allow better adaptation to medical terminology
        save_steps=500,  # Save model at every evaluation step
        logging_dir="./logs",
        logging_steps=5,  # Log at regular intervals to monitor training
        evaluation_strategy="steps",  # Evaluate periodically during training
        eval_steps=20,  # Evaluate every 20 steps
        save_total_limit=3,  # Keep only 3 most recent checkpoints to save disk space
        gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
        fp16=torch.cuda.is_available(),  # Use mixed precision for faster training
        dataloader_num_workers=4,  # Increase workers for faster data loading
        warmup_steps = 25,  # Proper warmup for stable training
        max_grad_norm=1.0,  # Prevent gradient explosion
        weight_decay=0.01,  # Regularization to reduce overfitting
        logging_strategy="steps",  # Log metrics at defined steps
        # predict_with_generate=True,  # Generate predictions during evaluation
        # report_to="tensorboard",  # Use TensorBoard for visualization
    )

    # Initialize trainer
    simple_logger = LoggerCallback(log_file="training_logs.json")

    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_dataset,  # Use encoded training data
        eval_dataset=encoded_test_dataset,    # Use encoded testing data
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=[simple_logger],
    )

    # Start training
    print("Starting fine-tuning...")
    trainer.train()

    # Save the model
    output_dir = "./whisper-finetuned"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
