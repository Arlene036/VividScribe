# prepare_data.py
import os
import json
import random
from pathlib import Path
import torchaudio
import numpy as np

def get_audio_files_with_answers(audio_dir, txt_mapper_path):
    # 加载txt_mapper
    with open(txt_mapper_path, 'r') as f:
        txt_mapper = json.load(f)
    
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                video_id = file.replace('.wav', '')
                
                # 检查是否在txt_mapper中
                if video_id in txt_mapper:
                    # 获取answer并计算词数
                    answer = txt_mapper[video_id][0]['question']  # 取第一个answer
                    word_count = len(answer.split())
                    
                    # 根据词数确定label
                    label = 1 if word_count >= 4 else 0
                    
                    audio_files.append({
                        'path': os.path.join(root, file),
                        'video_id': video_id,
                        'answer': answer,
                        'word_count': word_count,
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
    
    # 打印数据集统计信息
    def print_stats(name, dataset):
        total = len(dataset)
        label_0 = sum(1 for item in dataset if item['label'] == 0)
        label_1 = sum(1 for item in dataset if item['label'] == 1)
        print(f"{name} set:")
        print(f"Total: {total}")
        print(f"Label 0 (short answers): {label_0} ({label_0/total*100:.2f}%)")
        print(f"Label 1 (long answers): {label_1} ({label_1/total*100:.2f}%)")
        print()
    
    print_stats("Train", train_data)
    print_stats("Val", val_data)
    print_stats("Test", test_data)
    
    return train_data, val_data, test_data

def main():
    # 路径设置
    vast_dir = "/home/ubuntu/VividScribe/data/vast3k/audio_22050hz"
    volar_dir = "/home/ubuntu/VividScribe/data/volar3k/audio_22050hz"
    vast_mapper = "/home/ubuntu/VividScribe/data/vast3k/txt_mapper.json"
    volar_mapper = "/home/ubuntu/VividScribe/data/volar3k/txt_mapper.json"
    
    # 获取所有音频文件和对应的标签
    vast_files = get_audio_files_with_answers(vast_dir, vast_mapper)
    volar_files = get_audio_files_with_answers(volar_dir, volar_mapper)
    
    # 合并数据
    all_data = vast_files + volar_files
    
    # 打印总体统计信息
    total = len(all_data)
    label_0 = sum(1 for item in all_data if item['label'] == 0)
    label_1 = sum(1 for item in all_data if item['label'] == 1)
    print("Dataset statistics:")
    print(f"Total samples: {total}")
    print(f"Short answers (label 0): {label_0} ({label_0/total*100:.2f}%)")
    print(f"Long answers (label 1): {label_1} ({label_1/total*100:.2f}%)")
    print()
    
    # 分割数据集
    train_data, val_data, test_data = split_dataset(all_data)
    
    # 保存数据集
    dataset = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    output_path = 'dataset_split.json'
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()