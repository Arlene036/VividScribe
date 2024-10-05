"""Combine the valor60_non_verbal.json and vast_test60.json to create a new dataset"""

import json
import sys
import os
import shutil
from tqdm import tqdm

def combine_datasets(valor_path, vast_path, combined_path, mapping_path, valor_video_dir, vast_video_dir, combined_video_dir):
    # Load the input JSON files
    with open(valor_path, 'r') as valor_file:
        valor_data = json.load(valor_file)
    with open(vast_path, 'r') as vast_file:
        vast_data = json.load(vast_file)

    # rename fields in vast_data
    # clip_id -> video_id
    # vast_cap -> caption

    for item in vast_data:
        item['video_id'] = item.pop('clip_id')
        item['caption'] = item.pop('vast_cap')
        
    # rename fields in valor_data
    # desc -> caption
    
    for item in valor_data:
        item['caption'] = item.pop('desc')
        
    # create a mapping: 
    # {"vast": vast video ids, "valor_nonverbal": valor video ids}
    
    valor_ids = [item['video_id'] for item in valor_data]
    vast_ids = [item['video_id'] for item in vast_data]
    
    # extract video files
    for video_id in tqdm(valor_ids):
        video_path = os.path.join(valor_video_dir, f"{video_id}.mp4")
        # copy the video file to the combined video directory, if not exists, create the directory
        # create the directory if not exists
        # check if the file exists
        if not os.path.exists(combined_video_dir):
            os.makedirs(combined_video_dir)
        shutil.copy(video_path, os.path.join(combined_video_dir, f"{video_id}.mp4"))
    
    for video_id in tqdm(vast_ids):
        video_path = os.path.join(vast_video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist!")
            continue
        if not os.path.exists(combined_video_dir):
            os.makedirs(combined_video_dir)
        shutil.copy(video_path, os.path.join(combined_video_dir, f"{video_id}.mp4"))
    
    mapping = {
        "vast": valor_ids,
        "valor_nonverbal": vast_ids
    }
    
    with open(mapping_path, 'w') as mapping_file:
        json.dump(mapping, mapping_file, indent=4)
        
    with open(combined_path, 'w') as combined_file:
        json.dump(vast_data + valor_data, combined_file, indent=4)

    
if __name__ == "__main__":

    valor_ann_path = 'data/valor120/valor60_non_verbal.json'
    vast_ann_path = 'data/vast120/vast_test60.json'
    combined_ann_path = 'data/mix120/mix120.json'
    mapping_path = 'data/mix120/mapping.json'
    valor_video_dir = 'data/valor120/raw_video'
    vast_video_dir = 'data/vast120/raw_video'
    combined_video_dir = 'data/mix120/raw_video'
    
    # one-time execution
    # random sample 60 videos from vast_test120.json
    # import random
    # with open('data/vast120/vast_test120.json', 'r') as vast120:
    #     vast120_data = json.load(vast120)
    
    # # random sample 60 videos, excluding the video "NpktgLnCFP8.30
    # random.seed(42)
    # vast60_data = random.sample(vast120_data, 60)
    # with open('data/vast120/vast_test60.json', 'w') as vast60:
    #     json.dump(vast60_data, vast60, indent=4) 

    # Combine the datasets
    combine_datasets(valor_ann_path, vast_ann_path, combined_ann_path, mapping_path, valor_video_dir, vast_video_dir, combined_video_dir)
    print("Datasets combined successfully!")
    