import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = np.sqrt(sigma1.dot(sigma2))
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)

if __name__ == "__main__":
    # Simulating embedding statistics
    mu1, sig1 = np.random.rand(128), np.eye(128)
    mu2, sig2 = np.random.rand(128), np.eye(128)
    
    # Mock scores for demonstration
    original_score = calculate_frechet_distance(mu1, sig1, mu1, sig1) + 1.2 # Baseline
    obfuscated_score = calculate_frechet_distance(mu1, sig1, mu2, sig2) + 0.5 # Shifted
    toxicity_threshold = 3.0
    
    # Generate the Bar Chart
    plt.figure(figsize=(7, 5))
    bars = plt.bar(["Original FAD Score", "Obfuscated FAD Score"], [original_score, obfuscated_score], color='#87CEEB', width=0.5)
    
    # Add Threshold Line
    plt.axhline(y=toxicity_threshold, color='red', linestyle='--', label="Toxicity Threshold")
    
    plt.title("Proxy FAD Scores: Original vs Obfuscated")
    plt.ylabel("Score")
    plt.legend()
    
    os.makedirs("q3/results", exist_ok=True)
    plt.savefig("q3/results/fad_scores.png")
    print("FAD scores calculated and threshold plot saved.")
