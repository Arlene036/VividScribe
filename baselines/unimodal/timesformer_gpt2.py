import os
import sys
import json
import av
import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


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


if __name__ == "__main__":
    
    in_path = sys.argv[1]  # Path to the folder of video clips
    out_path = sys.argv[2]  # Path to the output json file
    
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

    gen_kwargs = {
            "min_length": 20, 
            "max_length": 50, 
            "num_beams": 12,
        }

    captions = []

    for video_dir in tqdm(video_dirs):
        file_path = os.path.join(in_path, video_dir)
        frames = extract_frames(file_path, model)
        # generate caption
        pixel_values = image_processor(frames, return_tensors="pt", padding=True).pixel_values.to(device)
        tokens = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        # print(caption)
        captions.append(caption)
    
        # save captions with video_ids to json file
        annotations = []
        
        for video_id, caption in zip(video_ids, captions):
            annotations.append({
                "video_id": video_id,
                "caption": caption
            })
            
        data = {
            "annotations": annotations
        }
        
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
    