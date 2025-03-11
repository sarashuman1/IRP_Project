import os
import librosa
import numpy as np
from datasets import Dataset, load_dataset
load_metric = load_dataset.load_metric  # Assign the function directly
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from pydub import AudioSegment
import io

# Load the WER metric
wer_metric = load_metric("wer")

def prepare_dataset(audio_dir, transcript_dir):
    audio_files = []
    transcripts = []
    
    print(f"Scanning directory: {audio_dir}")
    
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".mp3"):
            audio_file = os.path.join(audio_dir, file_name)
            transcript_file = os.path.join(transcript_dir, file_name.replace(".mp3", ".txt"))
            
            print(f"\nProcessing file: {file_name}")
            print(f"Audio path: {audio_file}")
            print(f"Transcript path: {transcript_file}")
            
            if not os.path.exists(audio_file):
                print(f"Error: Audio file does not exist: {audio_file}")
                continue
                
            if not os.path.exists(transcript_file):
                print(f"Error: Transcript file does not exist: {transcript_file}")
                continue
            
            try:
                # Print file size
                audio_size = os.path.getsize(audio_file)
                print(f"Audio file size: {audio_size} bytes")
                
                if audio_size == 0:
                    print(f"Error: Audio file is empty: {audio_file}")
                    continue
                
                # Load audio with pydub 
                audio_segment = AudioSegment.from_mp3(audio_file)
                audio_array = np.array(audio_segment.get_array_of_samples())
                audio = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                sr = audio_segment.frame_rate
                print(f"Loaded audio shape: {audio.shape}, Sample rate: {sr}")
                
                if len(audio.shape) == 0 or audio.size == 0:
                    print(f"Error: Audio data is empty after loading")
                    continue
                
                # Load and check transcript
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
                
                if not transcript:
                    print(f"Error: Transcript is empty: {transcript_file}")
                    continue
                    
                print(f"Successfully loaded audio and transcript")
                print(f"Transcript length: {len(transcript)} characters")
                
                audio_files.append(audio_file)
                transcripts.append(transcript)
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
    
    print(f"\nTotal valid pairs found: {len(audio_files)}")
    return {
        "audio": audio_files,
        "transcript": transcripts
    }

def preprocess_function(examples, processor):
    # Load and resample audio files
    max_length = 480000  # 30 seconds at 16kHz
    audio_arrays = []
    
    for audio_path in examples["audio"]:
        print(f"Preprocessing file: {audio_path}")
        audio_segment = AudioSegment.from_mp3(audio_path)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format='wav')
        wav_io.seek(0)
        audio, _ = librosa.load(io.BytesIO(wav_io.read()), sr=16000)
        
        # Handle length - either pad or truncate
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            padding = max_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        
        # Ensure audio is the correct shape (1, length)
        audio = audio.reshape(1, -1)
        audio_arrays.append(audio)

    # Process audio - each array should now be the same length
    try:
        audio_arrays = np.concatenate(audio_arrays, axis=0)
        print(f"Final audio array shape: {audio_arrays.shape}")
    except Exception as e:
        print("Error concatenating arrays:")
        for i, arr in enumerate(audio_arrays):
            print(f"Array {i} shape: {arr.shape}")
        raise e

    # Process audio
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Process text
    labels = processor.tokenizer(
        examples["transcript"],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    # Convert to numpy arrays
    input_features = inputs.input_features.numpy()
    label_arrays = labels.numpy()

    # Replace padding token id with -100 for training
    label_arrays[label_arrays == processor.tokenizer.pad_token_id] = -100

    return {
        "input_features": input_features,
        "labels": label_arrays
    }

# Data collator (moved outside main function)
class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Convert numpy arrays back to tensors
        input_features = torch.tensor([feature["input_features"] for feature in features])
        label_features = [torch.tensor(feature["labels"]) for feature in features] 

        # Pad labels to the maximum length in the batch
        label_features = torch.nn.utils.rnn.pad_sequence(
            label_features, batch_first=True, padding_value=-100
        )
        
        return {
            "input_features": input_features,
            "labels": label_features
        }

def compute_metrics(pred):
    """
    Computes metrics to track during fine-tuning.

    Args:
        pred (EvalPrediction): Predictions and labels from the Trainer.

    Returns:
        dict: A dictionary of metrics.
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode the predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate the WER (Word Error Rate)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def main():
    # Paths to dataset
    audio_path = "./audio"
    transcripts_path = "./transcripts"

    # Load model and processor first
    print("Loading model and processor...")
    model_name = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Prepare the dataset
    print("Preparing dataset...")
    data_dict = prepare_dataset(audio_path, transcripts_path)
    print(f"Found {len(data_dict['audio'])} valid audio-transcript pairs")

    # Create Dataset
    dataset = Dataset.from_dict(data_dict)

    # Process dataset in batches
    print("Processing dataset...")
    encoded_dataset = dataset.map(
        lambda examples: preprocess_function(examples, processor), # Pass processor here
        remove_columns=dataset.column_names,
        batch_size=8,
        batched=True
    )

    # Check if dataset is empty
    if len(encoded_dataset) == 0:
        raise ValueError("Dataset is empty! Please check the audio files and transcripts.")

    # Data collator
    data_collator = WhisperDataCollator(processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-finetuned",
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        num_train_epochs=3,
        save_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",  # Evaluate at each step
        eval_steps=100,              # Evaluate every 100 steps
        save_total_limit=2,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        logging_strategy="steps",
        warmup_steps=100,
        max_grad_norm=0.5,
        weight_decay=0.01,
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="wer",  # Use WER as the metric for best model selection
        greater_is_better=False,     # Lower WER is better
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,  # Pass the compute_metrics function
    )

    # Start training
    print("Starting fine-tuning...")
    trainer.train()

    # Save the model
    output_dir = "./whisper-finetuned"
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()