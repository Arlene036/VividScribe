"""
This file contains the evaluation of the caption regarding whether 
important objects are detected in the video or not. 
Rouge-1(unigram) recall score is used to calculate the overlap between 
the generated caption and the ground truth object labels.
"""

import json
import argparse
import os
import sys
from collections import Counter

GROUND_TRUTH_PATH = "output/mix120_groundtruth.json"
MAPPING_PATH = "data/mix120/mapping.json"

def load_mapping(mapping_file):
    """Load the verbal/non-verbal mapping file."""
    with open(mapping_file, 'r') as f:
        return json.load(f)

def filter_annotations(annotations, video_ids):
    """Filter annotations based on the given video IDs."""
    return [ann for ann in annotations if ann['video_id'] in video_ids]

def calculate_unigram_recall_single(groundtruth_labels, generated_caption):
    # Tokenize the labels and the caption
    groundtruth_tokens = groundtruth_labels.split()
    generated_tokens = generated_caption.split()
    
    # Count overlapping unigrams
    groundtruth_counter = Counter(groundtruth_tokens)
    generated_counter = Counter(generated_tokens)
    overlap = sum((groundtruth_counter & generated_counter).values())
    
    # Calculate recall
    total_groundtruth = sum(groundtruth_counter.values())
    recall = overlap / total_groundtruth if total_groundtruth > 0 else 0.0
    return recall

def calculate_unigram_recall(ground_truth, generated_captions, mapping, output_dir):
    """Calculate the recall score for the given captions."""
    
    # load ground truth and generated captions from json files to 
    print("Loading ground truth captions")
    
    with open(ground_truth, 'r') as f:
        ground_truth = json.load(f)
        
    with open(generated_captions, 'r') as f:
        generated_captions = json.load(f)

    results = {}
    
    for category in ['valor_nonverbal', 'vast']:
        print(f"Calculating recall for {category} videos")

        # Filter annotations
        gt_annotations = filter_annotations(ground_truth['annotations'], mapping[category])
        gen_annotations = filter_annotations(generated_captions['annotations'], mapping[category])

        # Calculate recall
        total = 0
        recall = 0
        
        for gt_ann, gen_ann in zip(gt_annotations, gen_annotations):
            # check if the video IDs match
            assert gt_ann['video_id'] == gen_ann['video_id']
            print("caption: ", gen_ann['caption'])
            print("object labels: ", gt_ann['object_labels'])
            total += 1
            recall += calculate_unigram_recall_single(gt_ann['object_labels'], gen_ann['caption'])
            
            # for light testing
            # if total == 5:
            #     break
        
        # around the recall score to 2 decimal places   
        avg_recall = round(recall / total, 2)
        
        # add the recall score to the results dictionary
        results[category] = {
            "rouge1_recall": avg_recall
        }
        
        print(f"Average unigram recall for {category} videos: {avg_recall}")
        
    # write the results to the output file
    with open(output_dir, 'w') as f:
        json.dump(results, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate unigram recall for object detection")
    parser.add_argument("--generated_cap_path", type=str, help="Path to the JSON file containing the generated captions.")
    parser.add_argument("--output_dir", type=str, help="Path to the output file to write the evaluation results.")    
    
    args = parser.parse_args()
    
    mapping = load_mapping(MAPPING_PATH)
    calculate_unigram_recall(GROUND_TRUTH_PATH, args.generated_cap_path, mapping, args.output_dir)