import torch
import torchaudio
import torchaudio.transforms as T

class PrivacyTransformer:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
    def obfuscate(self, waveform, n_steps=4.0):
        # Shifts pitch to mask age/gender while keeping linguistic content
        pitch_shift = T.PitchShift(self.sr, n_steps=n_steps)
        return pitch_shift(waveform)
