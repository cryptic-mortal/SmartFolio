import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import TemporalHGAT

def test_learning_capacity():
    print("--- 🧠 Testing Brain Learning Capacity (Distinguishable Signals) ---")
    
    # 1. Setup
    full_model = TemporalHGAT(num_stocks=1, input_dim=6, hidden_dim=64, num_heads=1)
    fusion_layer = full_model.sem_gat
    
    # Increase LR slightly for faster convergence in this toy problem
    optimizer = optim.Adam(fusion_layer.parameters(), lr=1e-6)
    
    print("Scenario: Advisor 0 (Truth) has a distinct pattern (Mean +2.0).")
    print("          Advisors 1-3 (Noise) are distinct (Mean -2.0).")
    print("Goal:     Model must learn to identify and trust the 'Positive' signal.\n")
    
    history = []
    
    for step in range(400): # Increased steps slightly
        # --- A. Generate Data ---
        # Ground Truth is random, but we give it a "fingerprint" (+2.0)
        # The model will learn: "Trust the vectors with positive values"
        raw_truth = torch.randn(1, 1, 64)
        ground_truth = raw_truth + 0.1 
        
        # Advisor 0: Sees the Truth (Shifted)
        input_0 = ground_truth.clone()
        
        # Advisor 1, 2, 3: See Noise (Shifted oppositely to be distinct)
        input_1 = torch.randn(1, 1, 64) - 0.1
        input_2 = torch.randn(1, 1, 64) - 0.1
        input_3 = torch.randn(1, 1, 64) - 0.1
        
        # Stack inputs: [Batch, Advisors=4, Stock, Hidden]
        stacked_inputs = torch.stack([input_0, input_1, input_2, input_3], dim=1)
        
        # --- B. Forward Pass ---
        # The Fusion Layer will try to pick the input that minimizes loss
        fused_output, weights = fusion_layer(stacked_inputs, require_weights=True)
        
        w = weights[0, :, 0, 0].detach().numpy()
        history.append(w)
        
        # --- C. Calculate Loss ---
        loss = torch.mean((fused_output - ground_truth) ** 2)
        
        # --- D. Backward Pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Weights: {np.round(w, 2)}")

    # 3. Verification
    final_weights = history[-1]
    print(f"\nFinal Weights: {np.round(final_weights, 4)}")
    
    # We expect Advisor 0 to be dominant (> 0.8)
    if final_weights[0] > 0.8:
        print("✅ PASS: The Brain successfully prioritized the Truth signal!")
    else:
        print("❌ FAIL: The Brain failed to converge.")

    # 4. Visualization
    history = np.array(history)
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 0], label='Advisor 0 (Truth)', linewidth=4, color='green')
    plt.plot(history[:, 1], label='Advisor 1 (Noise)', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.plot(history[:, 2], label='Advisor 2 (Noise)', linewidth=1.5, linestyle='--', alpha=0.7)
    plt.plot(history[:, 3], label='Advisor 3 (Noise)', linewidth=1.5, linestyle='--', alpha=0.7)
    
    plt.title("Learning Dynamics (Distinguishable Inputs)", fontsize=14)
    plt.xlabel("Training Steps")
    plt.ylabel("Trust Score (Attention Weight)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("brain_learning_proof.png")
    print("Chart saved to: brain_learning_proof.png")

if __name__ == "__main__":
    test_learning_capacity()