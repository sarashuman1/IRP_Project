import pandas as pd
import random

# Load the CSV data
def load_csv_data(filename):
    df = pd.read_csv(filename)
    return df['Phrase'].tolist()  # Convert phrases column to list

# Safely choose from a list or return a fallback
def safe_random_choice(choices, fallback="relevant information"):
    return random.choice(choices) if choices else fallback

# Generate a random sentence using phrases
def generate_sentence(phrases):
    patterns = [
        f"The doctor explains that {safe_random_choice(phrases)}",
        f"Medical findings indicate {safe_random_choice(phrases)}",
        f"Healthcare providers note that {safe_random_choice(phrases)}",
        f"Studies have shown that {safe_random_choice(phrases)}",
        f"It's important to understand that {safe_random_choice(phrases)}"
    ]
    return random.choice(patterns)

# Generate a paragraph of random sentences
def generate_text(num_sentences, phrases):
    return " ".join(generate_sentence(phrases) for _ in range(num_sentences))

# Main execution
def main():
    # Load phrases from CSV
    phrases = load_csv_data("IRP_dataset.csv")
    
    # Generate text files
    for i in range(1, 1000):  # Adjust range for number of files needed
        filename = f"medical_text_{i}.txt"
        # Generate approximately 300 words (about 15 sentences)
        content = generate_text(15, phrases)
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(content)
        print(f"Generated {filename}")

    print("\nAll files have been generated!")

if __name__ == "__main__":
    main()
