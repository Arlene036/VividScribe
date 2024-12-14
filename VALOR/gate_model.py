# model.py
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        gates = F.softmax(self.gate(x), dim=-1)  # [batch_size, num_experts]
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # [batch_size, num_experts, hidden_dim]

        output = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        return output, gates

class AudioTransformer(nn.Module):
    def __init__(self, 
                 n_mels=128,
                 hidden_dim=256,
                 num_layers=4,
                 num_heads=8,
                 num_experts=2,
                 dropout=0.1):
        super().__init__()
        
        # Mel spectrogram transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
            normalized=True
        )
        
        # Log-mel spectrogram
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Normalization layer
        self.spec_norm = nn.LayerNorm(n_mels)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Initial projection
        self.input_proj = nn.Linear(n_mels, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MoE layer
        self.num_experts = num_experts
        if num_experts > 1:
            self.moe = MoE(hidden_dim, hidden_dim*2, num_experts=num_experts)
        else:
            self.moe = None
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, 2)  # 2 classes
        
    def forward(self, x):
        # x shape: [batch_size, 1, time]
        batch_size = x.shape[0]
        
        # Convert to mel spectrogram
        with torch.no_grad():
            # Add a small constant to avoid log(0)
            x = self.mel_spec(x) + 1e-6  # [batch_size, n_mels, time]
            x = self.amplitude_to_db(x)   # Convert to dB scale
        
        # Reshape and transpose for processing

        x = x.squeeze(1)  # Remove the singleton dimension if it exists
        x = x.transpose(1, 2)  # [batch_size, time, n_mels]
        
        # Normalize
        x = self.spec_norm(x) # 
        
        # Project to hidden dimension
        x = self.input_proj(x)  # [batch_size, time, hidden_dim]
        
        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # MoE layer
        gates = None
        if self.num_experts > 1:
            x, gates = self.moe(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, gates

    def compute_mel_spec_length(self, audio_length):
        hop_length = 512
        return int((audio_length - 1024) / hop_length + 1)