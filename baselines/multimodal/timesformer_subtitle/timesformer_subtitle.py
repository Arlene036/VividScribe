"""
This multimodal baseline method 
(1) uses the Timesformer model to generate video captiond
(2) directly extract the subtitle from the video clips
(3) generate the caption by combining the subtitle and the video caption by few-shot learning using GPT-4o
"""

import os
import sys
import json
import av
import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
from openai import OpenAI
from dotenv import load_dotenv
import base64
import requests
import re
import json
from requests.exceptions import RequestException


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_IMAGE = "gpt-4o"
INSTRUCTION_POST_PROCESS = "The provided texts are (1) caption of a video clip generated from frames, and (2) ASR result of the audio of the video clip. If the ASR result has meaningful information, please use it to augment the caption and generate a high-level one sentence caption. Otherwise, please only output the caption of the video clip.\n\nCaption: {caption}\n\nASR: {asr}"

client = OpenAI(api_key=OPENAI_API_KEY)


# skip the following videos that failed to be processed
SKIPPED_IDS = [
    "p_o6NQX7lmE_0.000_10.000.mp4", 
    "xJ-6ewqMyxY_410.000_420.000.mp4", 
    "niJg7Q1XLyU_50.000_60.000.mp4", 
    "wj-gglKQ3KI_30.000_40.000.mp4"]


def extract_frames(video_dir, model):
    container = av.open(video_dir)

    # extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)

    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    
    # check if the number of frames is correct
    while len(frames) < clip_len:
        frames.append(frames[-1])
            
    return frames

def post_processing(image_caption, subtitle):
    completion = client.chat.completions.create(
        model=OPENAI_MODEL_IMAGE,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION_POST_PROCESS.format(caption=image_caption, asr=subtitle)
                    }
                ]
            }
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    
    in_path = sys.argv[1]  # Path to the folder of video clips
    in_ann_path = sys.argv[2]  # Path to the json file containing the subtitles
    out_path = sys.argv[3]  # Path to the output json file
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained processor, tokenizer, and model
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    # Load video data
    video_dirs = os.listdir(in_path)
    # skip the following videos
    video_dirs = [video_dir for video_dir in video_dirs if video_dir not in SKIPPED_IDS 
                  and video_dir.endswith(".mp4")]
    # sort video directories
    video_dirs.sort()
    print(video_dirs[0])
    print(f"Evaluating {len(video_dirs)} videos ...")
    # extract video ids, excluding the ".mp4" extension
    video_ids = [video_dir.split(".mp4")[0] for video_dir in video_dirs]
    
    with open(in_ann_path) as f:
        ann_data = json.load(f)
    
    # sort annotations by video_id
    subtitles = {ann["video_id"]: ann["subtitle"] for ann in ann_data}
    
    print("subtitle", subtitles)

    gen_kwargs = {
            "min_length": 20, 
            "max_length": 50, 
            "num_beams": 12,
        }

    vision_captions = []

    for video_dir in tqdm(video_dirs):
        file_path = os.path.join(in_path, video_dir)
        frames = extract_frames(file_path, model)
        # generate caption
        pixel_values = image_processor(frames, return_tensors="pt", padding=True).pixel_values.to(device)
        tokens = model.generate(pixel_values, **gen_kwargs)
        vision_caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        # print(caption)
        vision_captions.append(vision_caption)
        
    # post-processing to combine vision caption and subtitle
        
    annotations = []
    
    for video_id, vision_caption in tqdm(zip(video_ids, vision_captions)):
        subtitle = subtitles[video_id]
        
        # post-processing
        final_caption = post_processing(vision_caption, subtitle)
        
        annotations.append({
            "video_id": video_id,
            "caption": final_caption
        })
        
    data = {
        "annotations": annotations
    }
    
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    