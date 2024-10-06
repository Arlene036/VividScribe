"""
Generate clip captions using GPT-4o
Input: multiple frames from a video clip
       instruction text
Output: caption text
Note: use detail=low since all frames are low resolution. Each image costs a fixed 85 tokens.
"""

from openai import OpenAI
import os
import re
import base64
import requests
import sys
from dotenv import load_dotenv
import json
from tqdm import tqdm


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

INSTRUCTION = "The frames provided are sequentially sampled from the same video clip. Generate a caption for the video clip in one sentence."
OPENAI_MODEL = "gpt-4o"
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
                    "text": INSTRUCTION
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
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    caption = response['choices'][0]['message']['content']

    return caption


if __name__ == "__main__":
    in_path = sys.argv[1]  # Path to the folder of ultiple clips, where each clip is a folder of frames
    out_path = sys.argv[2]  # Path to the output json file

    captions = []
    for clip in tqdm(os.listdir(in_path)):
        if clip in SKIPPED_IDS:
            continue
        clip_path = os.path.join(in_path, clip)
        caption = generate_clip_caption(clip_path)
        captions.append({"video_id": clip, "caption": caption})

    captions_sorted = sorted(captions, key=lambda x: x["video_id"])
    output = {"annotations": captions_sorted}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)