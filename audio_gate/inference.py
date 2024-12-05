# inference.py
import torch
import torchaudio
from model import AudioTransformer
import json

def load_model(model_path):
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
    
    # Get prediction
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
    # Load test set
    with open('dataset_split.json', 'r') as f:
        dataset = json.load(f)
    test_data = dataset['test']
    
    # Load model
    model, device = load_model('best_model.pth')
    
    # Make predictions
    results = []
    for item in test_data:
        result = predict(model, item['path'], device)
        results.append({
            'path': item['path'],
            'true_label': item['label'],
            'predicted_label': result['prediction'],
            'confidence': result['confidence'],
            'expert_gates': result['expert_gates'].tolist()
        })
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()