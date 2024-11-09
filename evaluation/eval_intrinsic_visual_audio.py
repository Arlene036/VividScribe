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
    # Tokenize the labels and the caption, remove punctuation and convert to lowercase
    groundtruth_tokens = groundtruth_labels.lower().split()
    generated_tokens = generated_caption.lower().split()
    
    # remove punctuation
    groundtruth_tokens = [token.strip('.,') for token in groundtruth_tokens]
    generated_tokens = [token.strip('.,') for token in generated_tokens]
    
    # confirm that the caption is not empty
    if len(generated_tokens) == 0:
        return 0.0
    
    # Count overlapping unigrams
    groundtruth_counter = Counter(groundtruth_tokens)
    generated_counter = Counter(generated_tokens)
    overlap = sum((groundtruth_counter & generated_counter).values())
    
    # Calculate recall
    total_groundtruth = sum(groundtruth_counter.values())
    recall = overlap / total_groundtruth if total_groundtruth > 0 else 0.0
    return recall

def calculate_bert_score(references, candidates):
    """Calculate BERT score for a single caption pair."""
    P, R, F1 = score(candidates, references, lang="en", verbose=False)
    # convert tensors to lists
    P = P.tolist()
    R = R.tolist()
    F1 = F1.tolist()
    
    return P, R, F1


def calculate_metrics(
    annotation_type, 
    ground_truth, 
    generated_captions, 
    mapping, 
    output_dir, 
    individual_scores_path=None):
    """Calculate metrics for the specified annotation type (audio_cap or object_labels)."""
    
    # Load ground truth and generated captions
    print("Loading ground truth captions")
    with open(ground_truth, 'r') as f:
        ground_truth = json.load(f)
    with open(generated_captions, 'r') as f:
        generated_captions = json.load(f)

    results = {}
    detailed_scores = []

    if annotation_type == "audio_cap":
        for category in ['valor_nonverbal', 'vast']:
            print(f"Calculating metrics for {category} videos")
            
            # Filter annotations using mapping
            gt_annotations = filter_annotations(ground_truth['annotations'], mapping[category])
            gen_annotations = filter_annotations(generated_captions['annotations'], mapping[category])
            
            # extract 3 lists: groundtruth audio captions, generated captions, and video ids
            gt_audio_caps = [ann['audio_cap'] for ann in gt_annotations]
            gen_captions = [ann['caption'] for ann in gen_annotations]
            video_ids = [ann['video_id'] for ann in gt_annotations]
            
            # calculate bertscore for all pairs
            P, R, F1 = calculate_bert_score(gt_audio_caps, gen_captions)
            
            # get the metrics, convert tensors to lists
            metrics = {
                'rouge1_recalls': [],
                'bert_precisions': P, 
                'bert_recalls': R,
                'bert_f1s': F1
            }
            
            for i, (video_ids, gt, gen) in enumerate(zip(video_ids, gt_audio_caps, gen_captions)):

                rouge1_recall = calculate_unigram_recall_single(gt, gen)
                metrics['rouge1_recalls'].append(rouge1_recall)
                
                # check if the audio caption is empty
                if len(gt) == 0:
                    print(f"Empty audio caption for video: {video_ids}")

                # add other metrics to the detailed scores
                detailed_scores.append({
                    'video_id': video_ids,
                    'audiocap_groundtruth': gt,
                    'generated_caption': gen,
                    'audiocap_rouge1_recall': round(rouge1_recall, 4),
                    'audiocap_bert_precision': round(P[i], 4),
                    'audiocap_bert_recall': round(R[i], 4),
                    'audiocap_bert_f1': round(F1[i], 4)
                })
            
            results[category] = {
                "avg_audiocap_rouge1_recall": round(sum(metrics['rouge1_recalls']) / len(metrics['rouge1_recalls']), 4),
                "avg_audiocap_bert_precision": round(sum(metrics['bert_precisions']) / len(metrics['bert_precisions']), 4),
                "avg_audiocap_bert_recall": round(sum(metrics['bert_recalls']) / len(metrics['bert_recalls']), 4),
                "avg_audiocap_bert_f1": round(sum(metrics['bert_f1s']) / len(metrics['bert_f1s']), 4)
            }
    
    elif annotation_type == "object_labels":
        print("Calculating metrics for object_labels")
        
        video_ids = [ann['video_id'] for ann in ground_truth['annotations']]
        gt_object_labels = [ann['object_labels'] for ann in ground_truth['annotations']]
        gen_captions = [ann['caption'] for ann in generated_captions['annotations']]
        
        # Calculate BERT score for all pairs
        P, R, F1 = calculate_bert_score(gt_object_labels, gen_captions)
        
        # Evaluate across the entire test dataset without mapping
        metrics = {
            'rouge1_recalls': [],
            'bert_precisions': P,
            'bert_recalls': R,
            'bert_f1s': F1
        }

        for i, (vid, gt, gen) in enumerate(zip(video_ids, gt_object_labels, gen_captions)):

            recall = calculate_unigram_recall_single(gt, gen)

            metrics['rouge1_recalls'].append(recall)

            detailed_scores.append({
                'video_id': vid,
                'objectlabels_groundtruth': gt,
                'generated_caption': gen,
                'objectlabels_rouge1_recall': round(recall, 4),
                'objectlabels_bert_precision': round(P[i], 4),
                'objectlabels_bert_recall': round(R[i], 4),
                'objectlabels_bert_f1': round(F1[i], 4)
            })

        results = {
            "avg_objectlabels_rouge1_recall": round(sum(metrics['rouge1_recalls']) / len(metrics['rouge1_recalls']), 4),
            "avg_objectlabels_bert_precision": round(sum(metrics['bert_precisions']) / len(metrics['bert_precisions']), 4),
            "avg_objectlabels_bert_recall": round(sum(metrics['bert_recalls']) / len(metrics['bert_recalls']), 4),
            "avg_objectlabels_bert_f1": round(sum(metrics['bert_f1s']) / len(metrics['bert_f1s']), 4)
        }

    # Save results
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation Results saved to: {output_dir}")
    
    if individual_scores_path:
        os.makedirs(os.path.dirname(individual_scores_path), exist_ok=True)
        df = pd.DataFrame(detailed_scores)
        df.to_csv(individual_scores_path, index=False)
        print(f"\nDetailed scores saved to: {individual_scores_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intrinsic evaluation of the generated captions.")
    parser.add_argument("--annotation_type", choices=["audio_cap", "object_labels"], required=True, help="The annotation type to evaluate (audio_cap or object_labels).")
    parser.add_argument("--generated_cap_path", type=str, help="Path to the JSON file containing the generated captions.")
    parser.add_argument("--output_dir", type=str, help="Path to the output file to write the evaluation results.")
    parser.add_argument("--individual_scores_path", default=None, type=str, help="Path to the output file to write the detailed scores.")
    
    args = parser.parse_args()
    
    mapping = load_mapping(MAPPING_PATH) if args.annotation_type == "audio_cap" else None
    calculate_metrics(args.annotation_type, GROUND_TRUTH_PATH, args.generated_cap_path, mapping, args.output_dir, args.individual_scores_path)