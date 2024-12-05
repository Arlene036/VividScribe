# train.py
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import torchaudio
import numpy as np
from model import AudioTransformer
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AudioDataset(Dataset):
    def __init__(self, data_list, max_length=22050):
        self.data = data_list
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def pad_or_truncate(self, waveform):
        if waveform.size(1) > self.max_length:
            start = (waveform.size(1) - self.max_length) // 2
            waveform = waveform[:, start:start + self.max_length]
        elif waveform.size(1) < self.max_length:
            padding_length = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding_length))
        return waveform
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            waveform, sr = torchaudio.load(item['path'])
            if sr != 22050:
                waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)

            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = self.pad_or_truncate(waveform)
            
            return waveform, torch.tensor(item['label'], dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading file {item['path']}: {str(e)}")
            # 返回一个空音频和标签
            return torch.zeros((1, self.max_length)), torch.tensor(item['label'], dtype=torch.long)

def visualize_moe_gates(gates, labels, epoch, save_path=None):
    plt.figure(figsize=(10, 6))
    
    verbal_gates = gates[labels == 1].mean(axis=0)
    nonverbal_gates = gates[labels == 0].mean(axis=0)
    
    x = np.arange(len(verbal_gates))
    width = 0.35
    
    plt.bar(x - width/2, verbal_gates, width, label='Verbal', alpha=0.8)
    plt.bar(x + width/2, nonverbal_gates, width, label='Non-verbal', alpha=0.8)
    
    plt.xlabel('Expert Index')
    plt.ylabel('Average Gate Weight')
    plt.title(f'MoE Gate Weights Distribution (Epoch {epoch})')
    plt.xticks(x, [f'Expert {i+1}' for i in range(len(verbal_gates))])
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return {
        'verbal_gates': verbal_gates,
        'nonverbal_gates': nonverbal_gates
    }

def visualize_gate_heatmap(gates, labels, epoch, save_path=None):
    plt.figure(figsize=(12, 8))
    
    gate_matrix = np.array([
        gates[labels == 0].mean(axis=0),  # non-verbal
        gates[labels == 1].mean(axis=0)   # verbal
    ])
    
    sns.heatmap(gate_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=[f'Expert {i+1}' for i in range(gates.shape[1])],
                yticklabels=['Non-verbal', 'Verbal'],
                cmap='YlOrRd')
    
    plt.title(f'Expert Activation Heatmap (Epoch {epoch})')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return gate_matrix

def train_epoch(model, dataloader, criterion, optimizer, device, epoch,args):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    all_gates = []
    all_labels = []
    
    for waveforms, labels in tqdm(dataloader):
        waveforms, labels = waveforms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, gates = model(waveforms)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if gates is not None:
            all_gates.append(gates.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    gate_vis = None
    if all_gates:
        all_gates = np.concatenate(all_gates, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        os.makedirs('moe_vis', exist_ok=True)
        gate_vis = visualize_moe_gates(
            all_gates, 
            all_labels, 
            epoch,
            save_path=f'moe_vis/gates_epoch_{epoch}.png'
        )
    
        gate_heatmap = visualize_gate_heatmap(
            all_gates,
            all_labels,
            epoch,
            save_path=f'moe_vis/heatmap_epoch_{epoch}.png'
        )
    
    if not args.no_wandb:
        if gate_vis:
            wandb.log({
                'gate_distribution': wandb.Image(f'moe_vis/gates_epoch_{epoch}.png'),
                'gate_heatmap': wandb.Image(f'moe_vis/heatmap_epoch_{epoch}.png'),
                'expert1_verbal_weight': gate_vis['verbal_gates'][0],
                'expert2_verbal_weight': gate_vis['verbal_gates'][1],
                'expert1_nonverbal_weight': gate_vis['nonverbal_gates'][0],
                'expert2_nonverbal_weight': gate_vis['nonverbal_gates'][1],
            })
            
    return total_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, criterion, device, epoch, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_gates = []
    all_labels = []
    gate_vis = None
    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            logits, gates = model(waveforms)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if gates is not None:
                all_gates.append(gates.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    
    if all_gates:
        all_gates = np.concatenate(all_gates, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
 
        gate_vis = visualize_moe_gates(
        all_gates, 
        all_labels, 
        epoch,
        save_path=f'moe_vis/val_gates_epoch_{epoch}.png'
    )
    
        gate_heatmap = visualize_gate_heatmap(
            all_gates,
            all_labels,
            epoch,
            save_path=f'moe_vis/val_heatmap_epoch_{epoch}.png'
        )
    

    if not args.no_wandb:
        if gate_vis:
            wandb.log({
                    'val_gate_distribution': wandb.Image(f'moe_vis/val_gates_epoch_{epoch}.png'),
                'val_gate_heatmap': wandb.Image(f'moe_vis/val_heatmap_epoch_{epoch}.png'),
                'val_expert1_verbal_weight': gate_vis['verbal_gates'][0],
                'val_expert2_verbal_weight': gate_vis['verbal_gates'][1],
                'val_expert1_nonverbal_weight': gate_vis['nonverbal_gates'][0],
                'val_expert2_nonverbal_weight': gate_vis['nonverbal_gates'][1],
            })
    
    return total_loss / len(dataloader), 100 * correct / total

def get_args():
    parser = argparse.ArgumentParser(description='Audio Classification with MoE Training')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    
    # Model parameters
    parser.add_argument('--model-dim', type=int, default=256,
                        help='model hidden dimension (default: 256)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='number of transformer layers (default: 4)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='number of attention heads (default: 8)')
    parser.add_argument('--num-experts', type=int, default=2,
                        help='number of experts in MoE layer (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate (default: 0.1)')
    parser.add_argument('--num_experts', type=int, default=2,
                        help='number of experts in MoE layer (default: 2)')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default='dataset_split.json',
                        help='path to dataset split file (default: dataset_split.json)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loading workers (default: 4)')
    
    # Saving and logging
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='path to save outputs (default: outputs)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging (default: 10)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='how many epochs to wait before saving (default: 5)')
    parser.add_argument('--wandb-project', type=str, default='audio-classification-moe',
                        help='wandb project name (default: audio-classification-moe)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='wandb entity name (default: None)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='disable wandb logging')
    
    return parser.parse_args()

def main(args):
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    vis_dir = output_dir / 'moe_vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Load dataset splits
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    
    # Create datasets
    train_dataset = AudioDataset(dataset['train'])
    val_dataset = AudioDataset(dataset['val'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioTransformer(
        hidden_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        dropout=args.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, args
        )
        
        # Log metrics
        if not args.no_wandb:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'args': vars(args)
                },
                output_dir / 'best_model.pth'
            )
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'args': vars(args)
                },
                output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % args.log_interval == 0:
            print(f'Epoch {epoch+1}/{args.epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {best_val_acc:.2f}%')
            print('-' * 50)

if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
