import random
import json

# Load the JSON data
with open("medical_dataset_fully_merged.json", "r") as json_file:
    data = json.load(json_file)

# Safely choose from a list or return a fallback
def safe_random_choice(choices, fallback="relevant information"):
    return random.choice(choices) if choices else fallback

# Dynamically extract all phrases
def extract_all_phrases():
    phrases = []
    for category, subsets in data.items():
        if isinstance(subsets, dict):
            for subset, text_list in subsets.items():
                phrases.extend(text_list)
        elif isinstance(subsets, list):
            phrases.extend(subsets)
    return phrases

# Generate a random sentence using all text
def generate_sentence(phrases):
    patterns = [
        f"The doctor discusses {safe_random_choice(phrases)} in detail.",
        f"A key consideration is {safe_random_choice(phrases)}.",
        f"Patients often report {safe_random_choice(phrases)} during consultations.",
        f"Managing conditions involves understanding {safe_random_choice(phrases)}.",
        f"Research highlights the importance of {safe_random_choice(phrases)} in treatment."
    ]
    return random.choice(patterns)

# Generate a paragraph of random sentences
def generate_text(num_sentences, phrases):
    return " ".join(generate_sentence(phrases) for _ in range(num_sentences))

# Generate text files with all content dynamically
all_phrases = extract_all_phrases()

for i in range(1, 1000):  # Adjust range for the number of files you need
    filename = f"cardiology_text_{i}.txt"
    content = generate_text(5, all_phrases)  # Generate 5 sentences per file
    with open(filename, "w") as f:
        f.write(content)
    print(f"Generated {filename}")

print("\nAll files have been generated!")
