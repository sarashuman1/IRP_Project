import requests

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
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except FileNotFoundError:
        print(f"Error: Could not find the file {input_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    # Your API key
    API_KEY = "sk_aa83f4ff25a47027ab142a72134e2bd3b55c112c1f703c7e"
    
    # Get file names from user input
    input_file = input("Enter the input text file name: ")
    output_file = input("Enter the output MP3 file name: ")

    # Add .mp3 extension if not present in output filename
    if not output_file.lower().endswith('.mp3'):
        output_file += '.mp3'

    convert_text_to_speech(input_file, output_file, API_KEY)

if __name__ == "__main__":
    main()
