import json
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from eval_caption_tools.pycocoevalcap.eval import COCOEvalCap
from eval_caption_tools.pycocotools.coco import COCO


GROUND_TRUTH_PATH = "output/mix120_groundtruth.json"
MAPPING_PATH = "data/mix120/mapping.json"

def load_mapping(mapping_file):
    """Load the verbal/non-verbal mapping file."""
    with open(mapping_file, 'r') as f:
        return json.load(f)

def filter_annotations(annotations, video_ids):
    """Filter annotations based on the given video IDs."""
    return [ann for ann in annotations if ann['video_id'] in video_ids]

def evaluate_captions(generated_captions_file, ground_truth_file, mapping_file):
    """
    Evaluate the generated captions against ground-truth captions for verbal and non-verbal videos separately.

    Args:
        generated_captions_file (str): Path to the JSON file containing the generated captions.
        ground_truth_file (str): Path to the JSON file containing the ground-truth captions.
        mapping_file (str): Path to the JSON file containing the verbal/non-verbal mapping.
    
    Returns:
        dict: Evaluation metrics for verbal and non-verbal videos.
    """
    # Load ground-truth and generated captions
    print("Loading ground truth captions from {}".format(ground_truth_file))
    ground_truth = COCO(ground_truth_file)

    print("Loading generated captions from {}".format(generated_captions_file))
    generated_captions = ground_truth.loadRes(generated_captions_file)

    # Load verbal/non-verbal mapping
    print("Loading verbal/non-verbal mapping from {}".format(mapping_file))
    mapping = load_mapping(mapping_file)

    results = {}

    for category in ['valor_nonverbal', 'vast']:
        print(f"Evaluating {category} videos")
        
        # Filter annotations
        gt_annotations = filter_annotations(ground_truth.dataset['annotations'], mapping[category])
        gen_annotations = filter_annotations(generated_captions.dataset['annotations'], mapping[category])

        # Create new COCO objects with filtered annotations
        gt_filtered = COCO()
        gt_filtered.dataset['annotations'] = gt_annotations
        gt_filtered.createIndex()

        gen_filtered = COCO()
        gen_filtered.dataset['annotations'] = gen_annotations
        gen_filtered.createIndex()

        # Initialize the COCO caption evaluator
        coco_eval = COCOEvalCap(gt_filtered, gen_filtered)
        
        # Run the evaluation
        coco_eval.evaluate()
        
        eval_results = coco_eval.eval
        eval_results = {k: round(v*100, 2) for k, v in eval_results.items()}

        results[category] = eval_results
        print(f"Evaluation results for {category} videos: {eval_results}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated captions against ground-truth captions for verbal and non-verbal videos.")
    parser.add_argument("--generated_cap_path", type=str, help="Path to the JSON file containing the generated captions.")
    parser.add_argument("--output_dir", type=str, help="Path to the output file to write the evaluation results.")
    
    args = parser.parse_args()
    res = evaluate_captions(args.generated_cap_path, GROUND_TRUTH_PATH, MAPPING_PATH)

    # Write results to file
    with open(args.output_dir, 'w') as f:
        json.dump(res, f, indent=2)