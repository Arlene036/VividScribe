# prepare_data.py
import os
import json
import random
from pathlib import Path
import torchaudio
import numpy as np

def get_audio_files(data_dir, label):
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append({
                    'path': os.path.join(root, file),
                    'label': label
                })
    return audio_files

def split_dataset(data, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(data)
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def main():
    vast_dir = "/home/ubuntu/VividScribe/data/vast3k/audio_22050hz"
    volar_dir = "/home/ubuntu/VividScribe/data/volar3k/audio_22050hz"
    
    verbal_files = get_audio_files(vast_dir, label=1)
    nonverbal_files = get_audio_files(volar_dir, label=0)
    
    all_data = verbal_files + nonverbal_files
    
    train_data, val_data, test_data = split_dataset(all_data)

    dataset = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    with open('dataset_split.json', 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    main()