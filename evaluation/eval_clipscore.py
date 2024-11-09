import torch
import clip
from PIL import Image
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import argparse
import csv

class CLIPScoreCalculator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the CLIP Score Calculator.
        """
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def compute_clip_scores_for_example(self, text: str, frames_dir: str) -> tuple:
        """
        Compute CLIP scores for a single example (8 frames).
        """
        directory = Path(frames_dir)
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG'):
            image_paths.extend(directory.glob(ext))
        image_paths = sorted(image_paths)

        if not image_paths:
            return 0.0, 0.0

        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarity = (text_features @ image_features.T).item()
                scores.append(similarity)
                
            except Exception as e:
                continue

        if not scores:
            return 0.0, 0.0

        return np.mean(scores), np.max(scores)

    def process_dataset(self, json_path: str, frames_base_dir: str, detailed_output: str = None) -> tuple:
        """
        Process dataset and save detailed results if specified.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        all_means = []
        all_maxes = []
        
        # Open detailed output file if specified
        if detailed_output:
            csv_file = open(detailed_output, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['video_id', 'max_clipscore', 'mean_clipscore'])

        # Process each example
        for anno in tqdm(data['annotations'], desc="Processing examples"):
            video_id = anno['video_id']
            caption = anno['caption']
            frames_dir = Path(frames_base_dir) / video_id

            mean_score, max_score = self.compute_clip_scores_for_example(caption, str(frames_dir))
            all_means.append(mean_score)
            all_maxes.append(max_score)

            # Write to CSV if detailed output is specified
            if detailed_output:
                csv_writer.writerow([video_id, f"{max_score:.4f}", f"{mean_score:.4f}"])

        if detailed_output:
            csv_file.close()

        total_examples = len(data['annotations'])
        final_mean_score = np.sum(all_means) / total_examples
        final_max_score = np.sum(all_maxes) / total_examples

        return final_mean_score, final_max_score

def main():
    parser = argparse.ArgumentParser(description='Calculate CLIP scores for video frames')
    parser.add_argument('--json_path', required=True, help='Path to input JSON file')
    parser.add_argument('--detailed_output', help='Path to save detailed CSV output')
    parser.add_argument('--final_output', help='Path to save final scores')
    parser.add_argument('--frames_base_dir', default="data/mix120/extracted_data/frames_fps1",
                      help='Base directory for video frames')
    
    args = parser.parse_args()
    
    calculator = CLIPScoreCalculator()
    
    try:
        # Process dataset and get scores
        final_mean, final_max = calculator.process_dataset(
            args.json_path, 
            args.frames_base_dir,
            args.detailed_output
        )
        
        # Print to console
        print("\nFinal Results:")
        print(f"Average of mean scores: {final_mean:.4f}")
        print(f"Average of max scores: {final_max:.4f}")
        
        # Save final results if specified
        if args.final_output:
            with open(args.final_output, 'w') as f:
                f.write(f"Average of mean scores: {final_mean:.4f}\n")
                f.write(f"Average of max scores: {final_max:.4f}\n")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

if __name__ == "__main__":
    main()