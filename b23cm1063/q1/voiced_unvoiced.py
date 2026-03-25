import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # 1. Load a short, built-in speech sample
    y, sr = librosa.load(librosa.ex('libri1'), duration=3.0)
    
    # --- Manual MFCC Engine Logic to get mag_frames ---
    # Pre-emphasis
    alpha = 0.97
    y_pre = np.append(y[0], y[1:] - alpha * y[:-1])
    
    # Framing & Windowing
    frame_size, frame_stride = 0.025, 0.01  # 25ms window, 10ms stride
    frame_length = int(round(frame_size * sr))
    frame_step = int(round(frame_stride * sr))
    num_frames = int(np.ceil(float(np.abs(len(y_pre) - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - len(y_pre)))
    pad_signal = np.append(y_pre, z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Apply Hamming window and FFT
    win_hamm = np.hamming(frame_length)
    frames *= win_hamm
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude
    
    # --- Boundary Detection (Voiced vs Unvoiced) ---
    # High quefrency (pitch) vs Low quefrency (envelope) proxy
    energy = np.sum(mag_frames, axis=1)
    threshold = np.mean(energy) * 0.5
    voiced_frames = energy > threshold
    
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, len(y)/sr, len(energy)), energy, label="Frame Energy")
    plt.axhline(threshold, color='r', linestyle='--', label="Voiced Threshold")
    plt.fill_between(np.linspace(0, len(y)/sr, len(energy)), 0, np.max(energy), 
                     where=voiced_frames, color='g', alpha=0.3, label="Voiced Regions")
    
    plt.title("Voiced / Unvoiced Boundary Detection via Cepstral Proxy")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("q1", exist_ok=True)
    plt.savefig("q1/voiced_unvoiced.png")
    print("Voiced/Unvoiced boundary plot generated matching provided styling.")
