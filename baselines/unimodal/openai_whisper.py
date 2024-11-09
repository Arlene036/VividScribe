"""
Generate captions for all video clips using only ASR (Whisper) results
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import json
from tqdm import tqdm

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
OPENAI_MODEL_AUDIO = "whisper-1"
SKIPPED_IDS = ["p_o6NQX7lmE_0.000_10.000", "xJ-6ewqMyxY_410.000_420.000", 
               "niJg7Q1XLyU_50.000_60.000", "wj-gglKQ3KI_30.000_40.000"]

def generate_audio_transcript(path):
    """
    Generate transcript from audio file using Whisper
    """
    try:
        with open(path, "rb") as audio_file:
            transcription = client.audio.translations.create(
                model=OPENAI_MODEL_AUDIO, 
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        print(f"An error occurred while generating audio transcript for {path}: {e}")
        return None

def main(audio_folder, output_path):
    """
    Process all audio files and generate captions
    """
    # Get list of all audio files and sort them
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.wav')])
    captions = []

    # Process audio files
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Get video ID from filename (remove .wav extension)
        video_id = audio_file[:-4]
        
        # Skip if in SKIPPED_IDS
        if video_id in SKIPPED_IDS:
            continue

        # Generate transcript
        audio_path = os.path.join(audio_folder, audio_file)
        transcript = generate_audio_transcript(audio_path)

        if transcript:
            captions.append({
                "video_id": video_id,
                "caption": transcript
            })
        else:
            print(f"Failed to generate transcript for {video_id}")
            captions.append({
                "video_id": video_id,
                "caption": ""
            })

    # Prepare output
    output = {"annotations": captions}
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nProcessed {len(captions)} clips")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <audio_folder> <output_path>")
        sys.exit(1)

    audio_folder = sys.argv[1]  # Path to the folder containing audio files
    output_path = sys.argv[2]   # Path to the output json file

    main(audio_folder, output_path)