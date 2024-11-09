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
import torch
from bert_score import score
import pandas as pd

import warnings
import logging
from transformers import logging as trans_logging

warnings.filterwarnings("ignore")
trans_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

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

def calculate_bert_score_single(reference, candidate):
    """Calculate BERT score for a single caption pair."""
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return {
        "precision": P.item(),
        "recall": R.item(),
        "f1": F1.item()
    }

def calculate_metrics(ground_truth, generated_captions, mapping, output_dir, individual_scores_path=None):
    """Calculate the recall score and BERT score for the given captions."""
    
    # load ground truth and generated captions from json files to 
    print("Loading ground truth captions")
    
    with open(ground_truth, 'r') as f:
        ground_truth = json.load(f)
        
    with open(generated_captions, 'r') as f:
        generated_captions = json.load(f)

    results = {}
    detailed_scores = []

    for category in ['valor_nonverbal', 'vast']:
        print(f"Calculating recall for {category} videos")

        # Filter annotations
        gt_annotations = filter_annotations(ground_truth['annotations'], mapping[category])
        gen_annotations = filter_annotations(generated_captions['annotations'], mapping[category])

        metrics = {
            'rouge1_recalls': [],
            'bert_precisions': [],
            'bert_recalls': [],
            'bert_f1s': []
        }
        
        for gt_ann, gen_ann in zip(gt_annotations, gen_annotations):
            assert gt_ann['video_id'] == gen_ann['video_id']
            if len(gt_ann['object_labels']) == 0 or len(gt_ann['audio_cap'])==0:
              continue

            print("caption: ", gen_ann['caption'])
            print("object labels: ", gt_ann['object_labels'])

            recall = calculate_unigram_recall_single(gt_ann['object_labels'], gen_ann['caption'])
            bert_score = calculate_bert_score_single(gt_ann['audio_cap'], gen_ann['caption'])
            
            metrics['rouge1_recalls'].append(recall)
            metrics['bert_precisions'].append(bert_score['precision'])
            metrics['bert_recalls'].append(bert_score['recall'])
            metrics['bert_f1s'].append(bert_score['f1'])

            detailed_scores.append({
                'video_id': gt_ann['video_id'],
                'category': category,
                'audiocap_groundtruth': gt_ann['audio_cap'],
                'object_labels_groundtruth': gt_ann['object_labels'],
                'generated_caption': gen_ann['caption'],
                'rouge1_recall': round(recall, 4),
                'audiocap_bert_precision': round(bert_score['precision'], 4),
                'audiocap_bert_recall': round(bert_score['recall'], 4),
                'audiocap_bert_f1': round(bert_score['f1'], 4)
            })
            
            # for light testing
            # if len(detailed_scores) == 5:
            #     break
            # if len(detailed_scores) == 10:
            #     break
        
        results[category] = {
            "avg_rouge1_recall": round(sum(metrics['rouge1_recalls']) / len(metrics['rouge1_recalls']), 4),
            "avg_audiocap_bert_precision": round(sum(metrics['bert_precisions']) / len(metrics['bert_precisions']), 4),
            "avg_audiocap_bert_recall": round(sum(metrics['bert_recalls']) / len(metrics['bert_recalls']), 4),
            "avg_audiocap_bert_f1": round(sum(metrics['bert_f1s']) / len(metrics['bert_f1s']), 4)
        }
        
        print(f"\nScores for {category} videos:")
        for metric, value in results[category].items():
            print(f"- {metric}: {value}")
        
    # write the results to the output file
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEval AverageResults saved to: {output_dir}")
    
    if individual_scores_path:
        os.makedirs(os.path.dirname(individual_scores_path), exist_ok=True)
        df = pd.DataFrame(detailed_scores)
        df.to_csv(individual_scores_path, index=False)
        print(f"\nDetailed scores saved to: {individual_scores_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate unigram recall for object detection")
    parser.add_argument("--generated_cap_path", type=str, help="Path to the JSON file containing the generated captions.")
    parser.add_argument("--output_dir", type=str, help="Path to the output file to write the evaluation results.")
    parser.add_argument("--individual_scores_path", default=None, type=str, help="Path to the output file to write the detailed scores.")    
    
    args = parser.parse_args()
    
    mapping = load_mapping(MAPPING_PATH)
    calculate_metrics(GROUND_TRUTH_PATH, args.generated_cap_path, mapping, args.output_dir, args.individual_scores_path)