import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
import os
import time
import json
import librosa
import numpy as np

# Paths to the dataset
audio_dir = '/Users/hassanshuman/Local Training/Batch_MP3'
text_dir = '/Users/hassanshuman/Local Training/Batch_OUT'

# Load models and processors
original_model_name = "openai/whisper-tiny"
fine_tuned_model_path = '/Users/hassanshuman/Local Training/whisper-finetuned_2512'

processor = WhisperProcessor.from_pretrained(original_model_name)
original_model = WhisperForConditionalGeneration.from_pretrained(original_model_name)
fine_tuned_model = WhisperForConditionalGeneration.from_pretrained(fine_tuned_model_path)

def transcribe_and_calculate_loss(model, processor, audio_path, ground_truth):
    # Load and pad/truncate audio to match expected length
    audio, sr = librosa.load(audio_path, sr=16000)
    max_length = 16000 * 30  # 30 seconds
    if len(audio) < max_length:
        audio = np.pad(audio, (0, max_length - len(audio)))
    else:
        audio = audio[:max_length]
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = inputs.input_features
    labels = processor.tokenizer(ground_truth, return_tensors="pt", padding=True).input_ids

    model.eval()
    with torch.no_grad():
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss.item()
        predicted_ids = outputs.logits.argmax(dim=-1)
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription, loss
# Compare models
results = []

for file_name in os.listdir(audio_dir):
    if file_name.endswith(".mp3"):
        audio_path = os.path.join(audio_dir, file_name)
        text_path = os.path.join(text_dir, file_name.replace(".mp3", ".txt"))

        if not os.path.exists(text_path):
            print(f"Skipping {file_name}: No matching transcript found")
            continue

        with open(text_path, "r") as f:
            ground_truth = f.read().strip()

        # Evaluate Original Model
        start_time = time.time()
        original_transcription, original_loss = transcribe_and_calculate_loss(original_model, processor, audio_path, ground_truth)
        original_inference_time = time.time() - start_time
        original_wer = wer(ground_truth, original_transcription)

        # Evaluate Fine-Tuned Model
        start_time = time.time()
        fine_tuned_transcription, fine_tuned_loss = transcribe_and_calculate_loss(fine_tuned_model, processor, audio_path, ground_truth)
        fine_tuned_inference_time = time.time() - start_time
        fine_tuned_wer = wer(ground_truth, fine_tuned_transcription)

        # Store results
        results.append({
            "file_name": file_name,
            "ground_truth": ground_truth,
            "original": {
                "transcription": original_transcription,
                "wer": original_wer,
                "loss": original_loss,
                "inference_time": original_inference_time,
            },
            "fine_tuned": {
                "transcription": fine_tuned_transcription,
                "wer": fine_tuned_wer,
                "loss": fine_tuned_loss,
                "inference_time": fine_tuned_inference_time,
            },
        })

# Save results to a JSON file
output_file = "model_comparison_results_with_loss.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Comparison complete. Results saved to {output_file}")
