import json
import os
import torch
import subprocess
from pathlib import Path
import torchaudio
from model import AudioTransformer
from tqdm import tqdm

def load_audio_gate_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioTransformer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

def predict(model, audio_path, device):
    # Load and preprocess audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 22050:
        waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        logits, gates = model(waveform)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1)
        
    return {
        'prediction': prediction.item(),
        'confidence': probs.max().item(),
        'expert_gates': gates.cpu().numpy()
    }

def main():
    # Paths
    VIDEO_DIR = "/home/ubuntu/VividScribe/data/mix120/raw_video"
    AUDIO_DIR = "/home/ubuntu/VividScribe/data/mix120/extracted_data/audio_22050hz"
    MODEL_DIR = "/home/ubuntu/VividScribe/VALOR/output/VALOR_large" # Replace with actual model directory
    AUDIO_GATE_MODEL = "/home/ubuntu/VividScribe/audio_gate/output/best_model.pth"
    OUTPUT_FILE = "/home/ubuntu/VividScribe/output/competitive_valor.json"
    
    # Load data files
    with open("/home/ubuntu/VividScribe/data/mix120/mix120.json", "r") as f:
        mix120_data = json.load(f)
    with open("/home/ubuntu/VividScribe/output/unimodal_whisper_audiocap.json", "r") as f:
        captions = json.load(f)
    
    # Load audio gate model
    model, device = load_audio_gate_model(AUDIO_GATE_MODEL)
    
    # Results dictionary
    results = {
        "annotations": []
    }
    
    # Process each dataset
    for item in tqdm(mix120_data):
        video_id = item['video_id']
        subtitle = item.get('subtitle', '')  # Get subtitle from mix120 data
        
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        audio_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")
        
        # Check if files exist
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            print(f"Skipping {video_id}: files not found")
            continue
        
        ##### AUDIO GATE PREDICATION #####
        try:
            pred_result = predict(model, audio_path, device)
            
            # Select question based on prediction
            if pred_result['prediction'] == 0:  # non-verbal
                try:
                    question = f'The Audio transcript is "{captions[video_id]}". what is in the video'
                except KeyError:
                    print(f"Caption not found for {video_id}")
                    continue
            else:  # verbal
                question = f'The Audio transcript is "{subtitle}". what is in the video'
            
            # Run inference
            cmd = [
                "python", "inference.py",
                "--video_path", video_path,
                "--task", "qa%tva",
                "--model_dir", MODEL_DIR,
                "--question", question
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Add result to annotations in required format
            results["annotations"].append({
                "video_id": video_id,
                "caption": result.stdout.strip()  # 使用strip()去除多余的空白字符
            })
            
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
            continue
        
        # Save intermediate results every 10 videos
        if len(results["annotations"]) % 10 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=4)
    
    # Save final results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()