"""
Generate clip captions by 
    (1) captioning frames of a clip with gpt-4o
    (2) captioning audio of a clip with whisper-1
    (3) post-processing the captions
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import requests
import re
import sys
import json
from tqdm import tqdm
from requests.exceptions import RequestException


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

INSTRUCTION_IMAGE = "The frames provided are sequentially sampled from the same video clip. Generate a caption for the video clip in one sentence."
INSTRUCTION_POST_PROCESS = "The provided texts are (1) caption of a video clip generated from frames, and (2) ASR result of the audio of the video clip. If the ASR result has meaningful information (e.g. poeple are talking), please use it to augment the caption and generate a high-level one sentence caption. Otherwise, please only output the caption of the video clip.\n\nCaption: {caption}\n\nASR: {asr}"

OPENAI_MODEL_IMAGE = "gpt-4o"
OPENAI_MODEL_AUDIO = "whisper-1"

SKIPPED_IDS = ["p_o6NQX7lmE_0.000_10.000", "xJ-6ewqMyxY_410.000_420.000", "niJg7Q1XLyU_50.000_60.000", "wj-gglKQ3KI_30.000_40.000"]


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
  

def get_sorted_images(path):
    images = [f for f in os.listdir(path) if f.endswith(".jpg")]

    def get_frame_number(image):
        match = re.search(r'frame_(\d+)\.jpg', image)
        return int(match.group(1)) if match else 0

    sorted_images = sorted(images, key=get_frame_number)

    return [os.path.join(path, image) for image in sorted_images]


def generate_clip_caption(path):
    images = get_sorted_images(path)
    base64_images = [encode_image(image) for image in images]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": INSTRUCTION_IMAGE
                }
            ]
        }
    ]

    for base64_image in base64_images:
        messages[0]['content'].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                }
            }
        )

    payload = {
        "model": OPENAI_MODEL_IMAGE,
        "messages": messages,
        "max_tokens": 100
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except RequestException as e:
        print(f"An error occurred while generating clip caption: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        return None


def generate_audio_transcript(path):
    try:
        with open(path, "rb") as audio_file:
            transcription = client.audio.translations.create(
                model=OPENAI_MODEL_AUDIO, 
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        print(f"An error occurred while generating audio transcript: {e}")
        return None


def post_processing(image_caption, audio_caption):
    completion = client.chat.completions.create(
        model=OPENAI_MODEL_IMAGE,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION_POST_PROCESS.format(caption=image_caption, asr=audio_caption)
                    }
                ]
            }
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    frames_path = sys.argv[1]  # Path to the folder of ultiple clips, where each clip is a folder of frames
    audio_path = sys.argv[2]  # Path to the folder of audio files
    out_path = sys.argv[3]  # Path to the output json file

    captions = []
    for clip in tqdm(os.listdir(frames_path)):
        if clip in SKIPPED_IDS:
            continue
        clip_path = os.path.join(frames_path, clip)
        caption = generate_clip_caption(clip_path)
        transcript = generate_audio_transcript(os.path.join(audio_path, f"{clip}.wav"))
        final_caption = post_processing(caption, transcript)
        captions.append({"video_id": clip, "caption": final_caption})

    captions_sorted = sorted(captions, key=lambda x: x["video_id"])
    output = {"annotations": captions_sorted}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)