import json
import argparse
from eval_caption_tools.pycocoevalcap.eval import COCOEvalCap
from eval_caption_tools.pycocotools.coco import COCO

def evaluate_captions(generated_captions_file, ground_truth_file):
    """
    Evaluate the generated captions against ground-truth captions using the COCO caption evaluation tools.

    Args:
        generated_captions_file (str): Path to the JSON file containing the generated captions.
        ground_truth_file (str): Path to the JSON file containing the ground-truth captions.
        
        Both files should be in the COCO caption format:
        {"annotations": [{"video_id": int, "caption": str}]}
    
    Returns:
        dict: Evaluation metrics computed by the COCO caption evaluation tools.
    """

    # Load ground-truth and generated captions from their respective files
    print("Loading ground truth captions from {}".format(ground_truth_file))
    ground_truth = COCO(ground_truth_file)

    print("Loading generated captions from {}".format(generated_captions_file))
    generated_captions = ground_truth.loadRes(generated_captions_file)

    # Initialize the COCO caption evaluator
    print("Starting evaluation")
    coco_eval = COCOEvalCap(ground_truth, generated_captions)
    
    # Run the evaluation
    coco_eval.evaluate()
    
    eval_results = coco_eval.eval
    
    eval_results = {k: round(v*100, 2) for k, v in eval_results.items()}

    # Gather and return results
    print("Evaluation results: {}".format(eval_results))
    
    return eval_results

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate generated captions against ground-truth captions.")
    parser.add_argument("--generated_cap_path", type=str, help="Path to the JSON file containing the generated captions.")
    parser.add_argument("--true_cap_path", type=str, help="Path to the JSON file containing the ground-truth captions.")
    parser.add_argument("--output_dir", type=str, help="Path to the output file to write the evaluation results.")
    
    args = parser.parse_args()
    res = evaluate_captions(args.generated_cap_path, args.true_cap_path)

    # write results to file
    with open(args.output_dir, 'w') as f:
        json.dump(res, f)
    