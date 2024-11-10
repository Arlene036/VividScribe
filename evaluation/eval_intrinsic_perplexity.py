"""
This script evaluates the intrinsic perplexity of the generated captions using a pretrained Llama2 model.
"""

import argparse
import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

EXCLUDED_FILES = ["groundtruth.json", "mix120_groundtruth.json", "multimodal_llava-whisper-generation.json", "whisper_small_transcriptions.json"]

def evaluate_perplexity(model, tokenizer, captions, device):
    """
    Evaluate the perplexity of the given captions using the specified model and tokenizer.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for caption in tqdm(captions, desc="Evaluating perplexity"):
            if len(caption) == 0:
                continue
            inputs = tokenizer(caption, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids
            inputs = inputs.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item() * inputs.size(1)  # Multiply by the number of tokens in the sequence
            total_tokens += inputs.size(1)
    return torch.exp(torch.tensor(total_loss / total_tokens))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_cap_dir", type=str, required=True, help="Path to the directory containing the generated captions.")
    parser.add_argument("--model_name", type=str, default="/data/models/huggingface/meta-llama/Llama-2-7b-hf/", help="Name of the Llama2 model to use.")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to the result directory.")
    args = parser.parse_args()
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    # get all the files in the directory
    files = os.listdir(args.generated_cap_dir)
    files = [f for f in files if f not in EXCLUDED_FILES]
    
    print(f"Found {len(files)} files to evaluate.")

    # check the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the Llama2 model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for file in files:
        # Load the captions from the JSON file
        captions = []
        output_file = file.replace(".json", "_perplexity.json")
        
        if os.path.exists(os.path.join(args.result_dir, output_file)):
            print(f"Skipping {file} as the output file already exists.")
            continue
        
        with open(os.path.join(args.generated_cap_dir, file), "r") as f:
            data = json.load(f)
            for item in data["annotations"]:
                captions.append(item["caption"])

        if len(captions) == 0:
            raise ValueError("No captions found in the input JSON file.")

        # Evaluate the intrinsic perplexity
        perplexity = evaluate_perplexity(model, tokenizer, captions, device)
        print(f"Intrinsic perplexity for {file}: {perplexity.item()}")

        # Save the perplexity to a JSON file
        
        with open(os.path.join(args.result_dir, output_file), "w") as f:
            json.dump({"perplexity": perplexity.item()}, f, indent=4)