import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os

class PrivacyTransformer:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
    def obfuscate(self, waveform, n_steps=6.0):
        # Shifts pitch to obfuscate biometrics
        pitch_shift = T.PitchShift(self.sr, n_steps=n_steps)
        return pitch_shift(waveform)

def plot_spectrogram(ax, y, sr, title):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = ax.imshow(S_db, origin='lower', aspect='auto', cmap='magma')
    ax.set_title(title)
    ax.set_ylabel('Mel Bins')
    ax.set_xlabel('Frames')
    return img

if __name__ == "__main__":
    # Use real Librosa audio for a realistic spectrogram
    y, sr = librosa.load(librosa.ex('libri1'), sr=16000, duration=2.5)
    waveform = torch.tensor(y).unsqueeze(0)
    
    transformer = PrivacyTransformer(sample_rate=sr)
    obfuscated_wav = transformer.obfuscate(waveform, n_steps=6.0) 
    
    os.makedirs("q3/examples", exist_ok=True)
    torchaudio.save("q3/examples/original.wav", waveform, sr)
    torchaudio.save("q3/examples/obfuscated.wav", obfuscated_wav, sr)
    
    # --- Generate Spectrogram Visualizations ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # FIXED: Added .detach() to obfuscated_wav before converting to numpy
    plot_spectrogram(axes[0], waveform.squeeze().numpy(), sr, "Original Audio Spectrogram")
    plot_spectrogram(axes[1], obfuscated_wav.detach().squeeze().numpy(), sr, "Privacy-Preserved Audio (Biometrics Obfuscated)")
    
    plt.tight_layout()
    os.makedirs("q3/results", exist_ok=True)
    plt.savefig("q3/results/spectrogram_comparison.png")
    print("Privacy transformation applied and spectrograms saved.")
