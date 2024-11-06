"""
This script is used to select few-shot examples from the Valor and Vast datasets.
These examples should not be included in the mix120 dataset.
"""

import json
import random


def select_examples():
    vast120_path = "data/vast120/vast_test120.json"
    mix120_path = "data/mix120/mix120.json"
    valor32k_path = "data/valor32k/descs_cap_test.json"
    
    # randomly select 5 examples from the vast120 dataset that are not in the mix120 dataset
    with open(vast120_path, "r") as vast120_file:
        vast120_data = json.load(vast120_file)
        
    with open(mix120_path, "r") as mix120_file:
        mix120_data = json.load(mix120_file)
        
    with open(valor32k_path, "r") as valor32k_file:
        valor32k_data = json.load(valor32k_file)
        
    mix120_ids = [item["video_id"] for item in mix120_data]
    
    # select all valor32k examples which have an empty subtitle
    valor32k_nonverbal = [item for item in valor32k_data if not item.get("subtitle", "").strip()]
    
    # shuffle vast120_data and valor32k_nonverbal
    random.seed(42)
    random.shuffle(vast120_data)
    random.shuffle(valor32k_nonverbal)
    
    # select 5 examples from the vast120 dataset that are not in the mix120 dataset
    fewshot_examples = []
    
    for item in vast120_data:
        if item["clip_id"] not in mix120_ids:
            fewshot_examples.append(item)
            if len(fewshot_examples) == 5:
                break
    
    for item in valor32k_nonverbal:
        if item["video_id"] not in mix120_ids:
            fewshot_examples.append(item)
            if len(fewshot_examples) == 10:
                break
    
    # rename fields in fewshot_examples, clip_id -> video_id, vast_cap -> caption, desc -> caption
    for item in fewshot_examples:
        if "clip_id" in item:
            item["video_id"] = item.pop("clip_id")
        if "vast_cap" in item:
            item["caption"] = item.pop("vast_cap")
        if "desc" in item:
            item["caption"] = item.pop("desc")
    
    fewshot_path = "data/fewshot/fewshot_baseline.json"
    
    with open(fewshot_path, "w") as fewshot_file:
        json.dump(fewshot_examples, fewshot_file, indent=4)
    
    print(f"Saved few-shot examples to {fewshot_path}")
    

if __name__ == "__main__":
    select_examples()