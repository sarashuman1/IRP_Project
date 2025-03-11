import os
import json
import re
from jiwer import wer
import time

# Function to clean text by removing punctuation, special characters, and extra spaces
def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Paths to directories
original_text_dir = '/Users/hassanshuman/SaraIRPLocal/WER_TESTS/FINAL_TXT_TEST_ORIGINAL'  # Original transcripts
generated_text_dir = '/Users/hassanshuman/SaraIRPLocal/WER_TESTS/finetune_output'  # Generated transcripts from fine-tuned model

results = []

for file_name in os.listdir(original_text_dir):
    if file_name.endswith(".txt"):
        original_path = os.path.join(original_text_dir, file_name)
        generated_path = os.path.join(generated_text_dir, file_name)

        if not os.path.exists(generated_path):
            print(f"Skipping {file_name}: No matching generated transcript")
            continue

        with open(original_path, "r") as f:
            ground_truth = f.read().strip()
        
        with open(generated_path, "r") as f:
            generated_text = f.read().strip()

        # Clean the text before comparison
        ground_truth_cleaned = clean_text(ground_truth)
        generated_text_cleaned = clean_text(generated_text)

        wer_score = wer(ground_truth_cleaned, generated_text_cleaned)

        results.append({
            "file_name": file_name,
            "ground_truth": ground_truth,
            "generated": {
                "transcription": generated_text,
                "wer": wer_score
            }
        })

output_file = "/Users/hassanshuman/SaraIRPLocal/WER_TESTS/no_grammar_text_comparison_results_finetune.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Comparison complete. Results saved to {output_file}")
