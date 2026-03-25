# Audio Processing & Speech Representation Learning

**Roll Number:** B23CM1063  
**Dataset:** LibriSpeech (`train-clean-100`, `test-clean`)  

This repository contains the complete implementation for the Audio Processing and Speech Representation Learning assignment. It encompasses manual cepstral feature extraction, deep learning-based environment-agnostic speaker verification, and ethical fairness auditing for Automatic Speech Recognition (ASR) systems.

---

## 📂 Repository Structure

    📦 B23CM1063_Audio_Assignment
     ┣ 📜 README.md                  # This documentation
     ┣ 📜 requirements.txt           # Python dependencies
     ┣ 📂 q1/                        # Q1: Feature Extraction & Boundary Detection
     ┃ ┣ 📜 mfcc_manual.py           # Manual MFCC/Cepstrum Engine
     ┃ ┣ 📜 leakage_snr.py           # Spectral leakage & windowing analysis
     ┃ ┣ 📜 voiced_unvoiced.py       # Cepstral proxy boundary detection
     ┃ ┣ 📜 phonetic_mapping.py      # Wav2Vec2 forced alignment
     ┃ ┗ 📜 q1_report.pdf            # Methodology, RMSE table, and Q1 plots
     ┣ 📂 q2/                        # Q2: Environment-Agnostic Speaker Recognition
     ┃ ┣ 📂 configs/
     ┃ ┃ ┗ 📜 config.yaml            # Hyperparameters and architecture configs
     ┃ ┣ 📜 train.py                 # Dual-branch GRU training with Orthogonality Loss
     ┃ ┣ 📜 eval.py                  # Evaluation script (EER and ROC calculation)
     ┃ ┣ 📜 review.pdf               # Critical paper review & proposed GRL improvement
     ┃ ┣ 📜 q2_readme.md             # Specific instructions for Q2 reproduction
     ┃ ┗ 📂 results/                 # Training logs, checkpoints (.pth), t-SNE, and EER plots
     ┗ 📂 q3/                        # Q3: Ethical Auditing & Privacy
       ┣ 📜 audit.py                 # "Documentation Debt" and representation bias audit
       ┣ 📜 privacymodule.py         # Formant/Pitch shifting logic for biometrics
       ┣ 📜 pp_demo.py               # Generates original vs. obfuscated spectrograms
       ┣ 📜 train_fair.py            # Fairness-Aware loss function implementation
       ┣ 📂 evaluation_scripts/
       ┃ ┗ 📜 fad_proxy.py           # FAD Toxicity/Acceptability validation
       ┣ 📂 examples/                # Saved .wav files (original vs privacy-preserved)
       ┗ 📜 q3_report.pdf            # Ethical audit results, plots, and summary

---

## ⚙️ Setup & Installation

**1. Environment Setup** It is recommended to run this project in a virtual environment (or Kaggle/Colab notebook) with GPU support enabled for Question 2 and Question 3.

Install the required dependencies:
```bash
pip install -r requirements.txt

python q1/mfcc_manual.py
python q1/leakage_snr.py
python q1/voiced_unvoiced.py
python q1/phonetic_mapping.py
python q2/train.py
python q2/eval.py
python q3/audit.py
python q3/pp_demo.py
python q3/train_fair.py
python q3/evaluation_scripts/fad_proxy.py
