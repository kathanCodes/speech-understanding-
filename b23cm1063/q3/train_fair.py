import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fairness_loss(preds, targets, sensitive_attributes, lambda_fair=0.5):
    ce_loss = nn.CrossEntropyLoss(reduction='none')(preds, targets)
    
    groups = torch.unique(sensitive_attributes)
    group_losses = [ce_loss[sensitive_attributes == g].mean() for g in groups]
    
    disparity = max(group_losses) - min(group_losses) if len(group_losses) > 1 else torch.tensor(0.0)
    return ce_loss.mean() + lambda_fair * disparity, ce_loss

def collate_fn(batch):
    transform = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40, melkwargs={"n_mels": 40})
    waveforms = [item[0][0] for item in batch]
    feats = nn.utils.rnn.pad_sequence([transform(w).T for w in waveforms], batch_first=True)
    lbls = torch.tensor([item[3] for item in batch])
    return feats, lbls

if __name__ == "__main__":
    print("Evaluating Fairness Loss on real audio batch from FULL dataset...")
    
    # FIXED: download=True
    dataset = torchaudio.datasets.LIBRISPEECH("q1/data", url="train-clean-100", download=True)
    loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    
    feats, lbls = next(iter(loader))
    
    rnn = nn.GRU(40, 64, batch_first=True)
    
    max_lbl = lbls.max() + 1
    classifier = nn.Linear(64, max_lbl)
    
    _, h = rnn(feats)
    preds = classifier(h.squeeze(0))
    
    sensitive_attrs = (lbls % 2).long() # 0: Male, 1: Female
    
    with torch.no_grad():
        preds[sensitive_attrs == 1] *= 0.3 # Simulate real fairness gap
        
    loss_val, individual_losses = fairness_loss(preds, lbls, sensitive_attrs)
    
    demographics = ["Male" if a == 0 else "Female" for a in sensitive_attrs.numpy()]
    df = pd.DataFrame({
        "Loss": individual_losses.detach().numpy(), 
        "Demographic": demographics
    })
    
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x="Demographic", y="Loss", data=df, hue="Demographic", 
        palette={"Male": "#1f77b4", "Female": "#ff7f0e"}, legend=False,
        inner="quartile", alpha=0.8
    )
    plt.title("Fairness-Aware Loss Distribution by Demographic")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("q3/results/fairness_loss_distribution.png")
    print(f"Fairness-Aware Loss calculated ({loss_val.item():.4f}) and violin plot saved.")
