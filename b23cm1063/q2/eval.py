import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

class DisentangledSpeakerNet(nn.Module):
    def __init__(self, n_mfcc=40, num_speakers=300): 
        super().__init__()
        self.rnn = nn.GRU(n_mfcc, 128, batch_first=True)
        self.speaker_encoder = nn.Linear(128, 64)
        self.env_encoder = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_speakers) 
        
    def forward(self, x):
        _, h = self.rnn(x) 
        h = h.squeeze(0)
        s_emb = torch.relu(self.speaker_encoder(h))
        e_emb = torch.relu(self.env_encoder(h))
        return s_emb, e_emb, self.classifier(s_emb)

def collate_fn(batch):
    transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={"n_mels": 40})
    waveforms = [item[0][0] for item in batch]
    feats = nn.utils.rnn.pad_sequence([transform(w).T for w in waveforms], batch_first=True)
    lbls = torch.tensor([item[3] for item in batch]) 
    return feats, lbls

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer, fpr, tpr

if __name__ == "__main__":
    print("Starting Q2 Evaluation on Test Set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DisentangledSpeakerNet().to(device)
    model_path = "q2/results/disentangled_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model checkpoint loaded successfully.")
    else:
        print("Warning: Model checkpoint not found.")
        
    model.eval()
    test_dataset = torchaudio.datasets.LIBRISPEECH("q1/data", url="test-clean", download=True)
    test_loader = DataLoader(torch.utils.data.Subset(test_dataset, range(200)), batch_size=64, collate_fn=collate_fn)
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for feats, lbls in test_loader:
            feats = feats.to(device)
            s_emb, _, _ = model(feats)
            all_embeddings.append(s_emb.cpu())
            all_labels.append(lbls)
            
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    
    scores, pair_labels = [], []
    for i in range(len(all_embeddings)):
        for j in range(i + 1, len(all_embeddings)):
            sim = torch.nn.functional.cosine_similarity(all_embeddings[i].unsqueeze(0), all_embeddings[j].unsqueeze(0))
            scores.append(sim.item())
            pair_labels.append(1 if all_labels[i] == all_labels[j] else 0)
            
    eer, fpr, tpr = compute_eer(pair_labels, scores)
    print(f"Evaluation Complete. Equal Error Rate (EER): {eer:.4f}")
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (EER = {eer:.3f})')
    plt.plot([0, 1], [1, 0], color='red', lw=1, linestyle='--', label='EER Line')
    plt.title("Speaker Verification Performance on Test Set")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("q2/results/evaluation_eer_curve.png")
