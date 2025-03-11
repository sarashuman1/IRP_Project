import requests
import os
from time import sleep

def convert_text_to_speech(input_file, output_file, api_key):
    try:
        # Read the text file
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # API endpoint
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"  # Adam's voice ID
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Audio successfully generated and saved as {output_file}")
            return True
        else:
            print(f"Error processing {input_file}: {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"An error occurred processing {input_file}: {str(e)}")
        return False

def process_files():
    # Your API key
    API_KEY = "sk_7b3101b70a62b0f5aaaabbb3fbe6350ed130703c53cce9ec"
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Create output folder for MP3s if it doesn't exist
    output_folder = os.path.join(current_dir, "mp3_output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all .txt files in current directory
    txt_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.txt')]
    
    if not txt_files:
        print("No .txt files found in the current directory.")
        return
    
    print(f"Found {len(txt_files)} text files to process.")
    
    # Process each file
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\nProcessing file {i} of {len(txt_files)}: {txt_file}")
        
        input_path = os.path.join(current_dir, txt_file)
        output_path = os.path.join(output_folder, txt_file.replace('.txt', '.mp3'))
        
        # Convert the file
        success = convert_text_to_speech(input_path, output_path, API_KEY)
        
        # Add a delay between API calls to avoid rate limiting
        if success and i < len(txt_files):
            print("Waiting 2 seconds before next file...")
            sleep(2)

if __name__ == "__main__":
    process_files()

