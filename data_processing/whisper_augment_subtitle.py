"""
Relabel/Augment subtitle using whisper-1
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from tqdm import tqdm


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_MODEL_AUDIO = "whisper-1"

SKIPPED_IDS = ["p_o6NQX7lmE_0.000_10.000", "xJ-6ewqMyxY_410.000_420.000", "niJg7Q1XLyU_50.000_60.000", "wj-gglKQ3KI_30.000_40.000"]


def generate_audio_transcript(path):
    audio_file= open(path, "rb")
    transcription = client.audio.translations.create(
        model=OPENAI_MODEL_AUDIO, 
        file=audio_file
    )
    return transcription.text


if __name__ == "__main__":
    in_path = "data/test120/sample_v_nv_test120_new.json"
    out_path = "data/test120/sample_v_nv_test120_new_whisper.json"
    audio_path = "data/test120/extracted_data/audio_22050hz"

    with open(in_path, "r") as f:
        data = json.load(f)

    output = []
    for item in tqdm(data):
        video_id = item["video_id"]
        if video_id in SKIPPED_IDS:
            continue
        desc = item["desc"]
        audio_file = f"{audio_path}/{video_id}.wav"
        original_transcript = item["subtitle"]
        if original_transcript == "":
            output.append({"video_id": video_id, "desc": desc, "subtitle": ""})
        else:
            transcript = generate_audio_transcript(audio_file)
            output.append({"video_id": video_id, "desc": desc, "subtitle": transcript})

    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)