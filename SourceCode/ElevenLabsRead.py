import requests

# Your API key
API_KEY = "x"

# File paths
input_file = 'cardiofullletter.txt'
output_file = 'cardio_full_letter.mp3'

try:
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # API endpoint
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"  # Adam's voice ID
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY
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
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except FileNotFoundError:
    print(f"Error: Could not find the file {input_file}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
