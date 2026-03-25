# Q2: Disentangled Representation Learning for Environment-Agnostic Speaker Recognition

## Overview
This folder contains the implementation of a dual-branch Recurrent Neural Network (GRU) designed to disentangle speaker identity from environmental factors using an Orthogonality Loss penalty, evaluated on the LibriSpeech dataset.

## How to Reproduce the Experiments
1. **Environment Setup:** Ensure all dependencies from the root `requirements.txt` are installed (specifically `torch`, `torchaudio`, and `matplotlib`).
2. **Dataset:** The script automatically verifies and utilizes the `train-clean-100` split of the LibriSpeech corpus. No manual downloading is required; `torchaudio` will handle the cache.
3. **Training & Evaluation:** Navigate to the root directory of the repository and execute:
   ```bash
   python q2/train.py
