from openai import OpenAI
import os
import re
import base64
import requests
import sys
from dotenv import load_dotenv
import json
from tqdm import tqdm
import time
import math

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

INSTRUCTION = "The frames provided are sequentially sampled from the same video clip. Generate a caption for the video clip in one sentence."
OPENAI_MODEL = "gpt-4o"
SKIPPED_IDS = ["p_o6NQX7lmE_0.000_10.000", "xJ-6ewqMyxY_410.000_420.000", "niJg7Q1XLyU_50.000_60.000", "wj-gglKQ3KI_30.000_40.000"]
BATCH_SIZE = 4  # Process 4 examples before waiting
SLEEP_TIME = 20  # Sleep time between batches
MAX_RETRIES = 3

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

def handle_rate_limit(error_message):
    """Extract wait time from rate limit error message and wait"""
    try:
        wait_time = float(re.search(r'Please try again in (\d+\.?\d*)s', error_message).group(1))
        wait_time = math.ceil(wait_time + 1)
        print(f"\nRate limit reached. Waiting for {wait_time} seconds...")
        time.sleep(wait_time)
        return True
    except (AttributeError, ValueError):
        return False

def generate_clip_caption(path, retry_count=0):
    if retry_count >= MAX_RETRIES:
        raise Exception(f"Failed after {MAX_RETRIES} retries")
    
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
                    "detail": "high"
                }
            }
        )

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_tokens": 100
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, 
                               json=payload).json()
        
        if 'error' in response:
            error_msg = response['error'].get('message', '')
            if 'rate_limit' in error_msg.lower():
                if handle_rate_limit(error_msg):
                    return generate_clip_caption(path, retry_count + 1)
            raise Exception(f"API Error: {error_msg}")
            
        return response['choices'][0]['message']['content']
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if retry_count < MAX_RETRIES:
            print(f"Retrying... (attempt {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(SLEEP_TIME)
            return generate_clip_caption(path, retry_count + 1)
        raise

def process_batch(clips_batch, in_path):
    """Process a batch of clips and return their captions"""
    batch_results = []
    batch_failures = []
    
    for clip in clips_batch:
        clip_path = os.path.join(in_path, clip)
        try:
            caption = generate_clip_caption(clip_path)
            batch_results.append({"video_id": clip, "caption": caption})
        except Exception as e:
            print(f"\nFailed to process clip {clip}: {str(e)}")
            batch_failures.append({"video_id": clip, "error": str(e)})
    
    return batch_results, batch_failures

if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # Get list of clips to process
    clips = [clip for clip in os.listdir(in_path) if clip not in SKIPPED_IDS]
    
    captions = []
    failed_clips = []
    
    # Process clips in batches
    for i in tqdm(range(0, len(clips), BATCH_SIZE)):
        batch_clips = clips[i:i + BATCH_SIZE]
        
        # Process current batch
        batch_results, batch_failures = process_batch(batch_clips, in_path)
        captions.extend(batch_results)
        failed_clips.extend(batch_failures)
        
        # Sleep after each batch (except for the last one)
        if i + BATCH_SIZE < len(clips):
            print(f"\nProcessed {i + len(batch_clips)}/{len(clips)} clips. Sleeping for {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)

    # Sort and save results
    captions_sorted = sorted(captions, key=lambda x: x["video_id"])
    output = {
        "annotations": captions_sorted,
        "failed_clips": failed_clips
    }
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    if failed_clips:
        print(f"\nFailed to process {len(failed_clips)} clips. Check the output file for details.")
    
    print(f"\nSuccessfully processed {len(captions)} clips!")