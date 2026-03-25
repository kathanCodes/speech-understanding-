import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import os
from tqdm import tqdm

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

def orthogonality_loss(s_emb, e_emb):
    s_norm = s_emb / (s_emb.norm(dim=1, keepdim=True) + 1e-8)
    e_norm = e_emb / (e_emb.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mean((s_norm * e_norm).sum(dim=1)**2)

def collate_fn(batch):
    transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={"n_mels": 40})
    waveforms = [item[0][0] for item in batch]
    feats = nn.utils.rnn.pad_sequence([transform(w).T for w in waveforms], batch_first=True)
    raw_lbls = torch.tensor([item[3] for item in batch]) 
    return feats, raw_lbls

if __name__ == "__main__":
    print("Loading FULL LibriSpeech Dataset for Q2 Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = torchaudio.datasets.LIBRISPEECH("q1/data", url="train-clean-100", download=True)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    
    model = DisentangledSpeakerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ce_loss_fn = nn.CrossEntropyLoss()
    
    epochs = 15
    acc_history, ortho_history = [], []
    speaker_map = {}
    
    model.train()
    for epoch in range(epochs):
        epoch_ortho = 0
        correct, total = 0, 0
        
        for feats, raw_lbls in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            feats = feats.to(device)
            
            # Map random IDs to sequential IDs safely
            mapped_lbls = []
            for l in raw_lbls.tolist():
                if l not in speaker_map:
                    speaker_map[l] = len(speaker_map)
                mapped_lbls.append(speaker_map[l])
            lbls = torch.tensor(mapped_lbls).to(device)
            
            optimizer.zero_grad()
            s_emb, e_emb, preds = model(feats)
            
            max_lbl = lbls.max() + 1
            if preds.size(1) < max_lbl:
                pad_size = max_lbl - preds.size(1)
                preds = torch.cat([preds, torch.zeros(preds.size(0), pad_size).to(device)], dim=1)
                
            loss_ce = ce_loss_fn(preds, lbls)
            loss_ortho = orthogonality_loss(s_emb, e_emb)
            loss = loss_ce + 0.1 * loss_ortho
            
            loss.backward()
            optimizer.step()
            
            epoch_ortho += loss_ortho.item()
            correct += (preds.argmax(1) == lbls).sum().item()
            total += lbls.size(0)
            
        acc_history.append((correct / total) * 100)
        ortho_history.append(epoch_ortho / len(loader))

    os.makedirs("q2/results", exist_ok=True)

    # 1. Save standard Training Metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, epochs+1), acc_history, marker='o', color='#2ca02c')
    axes[0].set_title("Speaker Recognition Accuracy vs Epochs")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True)
    axes[1].plot(range(1, epochs+1), ortho_history, marker='s', color='#9467bd')
    axes[1].set_title("Orthogonality Loss over Time")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    plt.savefig("q2/results/training_metrics.png")
    
    # 2. Save the Trained Model Checkpoint
    torch.save(model.state_dict(), "q2/results/disentangled_model.pth")
    
    # 3. Save CSV Training Logs
    df_logs = pd.DataFrame({"Epoch": range(1, epochs+1), "Accuracy": acc_history, "Orthogonality_Loss": ortho_history})
    df_logs.to_csv("q2/results/training_logs.csv", index=False)
    
    # 4. Generate t-SNE Visualizations
    model.eval()
    with torch.no_grad():
        feats, raw_lbls = next(iter(loader))
        feats = feats.to(device)
        s_emb, e_emb, preds = model(feats)
        
        s_np = s_emb.cpu().numpy()
        e_np = e_emb.cpu().numpy()
        lbls_np = raw_lbls.numpy()

        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        s_tsne = tsne.fit_transform(s_np)
        e_tsne = tsne.fit_transform(e_np)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(s_tsne[:, 0], s_tsne[:, 1], c=lbls_np, cmap='tab10', alpha=0.7)
        axes[0].set_title("t-SNE: Speaker Embeddings\n(Clustered by Identity)")
        axes[1].scatter(e_tsne[:, 0], e_tsne[:, 1], c=lbls_np, cmap='tab10', alpha=0.7)
        axes[1].set_title("t-SNE: Environment Embeddings\n(Mixed Cloud)")
        plt.tight_layout()
        plt.savefig("q2/results/tsne_embeddings.png")
        
        # 5. Generate ROC Curve
        probs = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
        mapped_eval_lbls = torch.tensor([speaker_map[l.item()] for l in raw_lbls]).numpy()
        true_binary = (preds.argmax(1).cpu().numpy() == mapped_eval_lbls).astype(int)
        pred_scores = probs.max(axis=1)
        
        fpr, tpr, _ = roc_curve(true_binary, pred_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Training Batch Verification ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig("q2/results/roc_curve.png")
        
    print("Q2 Training and Artifact Generation Complete!")
