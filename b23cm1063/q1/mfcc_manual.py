import numpy as np
import librosa
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def manual_mfcc(y, sr, n_mfcc=13, n_fft=512, hop_length=160, n_mels=40):
    # 1. Pre-emphasis
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # 2. Windowing (Hamming)
    frames = np.lib.stride_tricks.sliding_window_view(y_pre, n_fft)[::hop_length] * np.hamming(n_fft)
    
    # 3. FFT & Power Spectrum
    pow_frames = (1.0 / n_fft) * (np.absolute(np.fft.rfft(frames, n_fft)) ** 2)
    
    # 4. Mel-Filterbank
    mel_points = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    
    for m in range(1, n_mels + 1):
        for k in range(bin[m-1], bin[m]): fbank[m-1, k] = (k - bin[m-1]) / (bin[m] - bin[m-1])
        for k in range(bin[m], bin[m+1]): fbank[m-1, k] = (bin[m+1] - k) / (bin[m+1] - bin[m])
            
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    
    # 5. Log-compression & DCT
    mfcc = dct(20 * np.log10(filter_banks), type=2, axis=1, norm='ortho')[:, 1 : (n_mfcc + 1)]
    return mfcc.T

if __name__ == "__main__":
    y, sr = librosa.load(librosa.ex('libri1'), sr=16000)
    mfccs = manual_mfcc(y, sr)
    plt.imshow(mfccs, aspect='auto', origin='lower')
    plt.title("Manual MFCCs")
    plt.savefig("q1/manual_mfcc_plot.png")
    print("MFCCs extracted and saved.")
