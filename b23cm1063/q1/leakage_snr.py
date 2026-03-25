import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # 1. Load a short, built-in speech sample
    y, sr = librosa.load(librosa.ex('libri1'), duration=3.0)
    
    # Frame settings from your provided code
    frame_size = 0.025  # 25ms window
    frame_length = int(round(frame_size * sr))
    
    # Spectral Leakage & Windowing Analysis
    win_rect = np.ones(frame_length)
    win_hamm = np.hamming(frame_length)
    win_hann = np.hanning(frame_length)
    
    plt.figure(figsize=(12, 4))
    plt.title("Spectral Leakage Analysis: Window Functions (Time Domain)")
    plt.plot(win_rect, label="Rectangular")
    plt.plot(win_hamm, label="Hamming")
    plt.plot(win_hann, label="Hanning")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("q1", exist_ok=True)
    plt.savefig("q1/spectral_leakage_windows.png")
    print("Spectral leakage plot generated matching provided styling.")
