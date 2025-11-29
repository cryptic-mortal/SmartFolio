import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import TemporalHGAT

def setup_model(num_stocks=20, hidden_dim=64):
    """Init model with fixed seed."""
    torch.manual_seed(42)
    model = TemporalHGAT(
        num_stocks=num_stocks,
        input_dim=6,
        lookback=30,
        hidden_dim=hidden_dim,
        num_heads=4
    )
    model.eval()
    return model

def test_generic_masking(num_stocks=20, density=0.3):
    print(f"--- 🧪 Generic GAT Masking Test (N={num_stocks}, Density={density}) ---")
    
    # 1. Setup Model
    model = setup_model(num_stocks)
    
    # 2. Generate Random Graph (Adjacency Matrix)
    # 1 = Connected, 0 = Blocked
    # We enforce self-loops (diagonal = 1)
    rand_matrix = torch.rand(num_stocks, num_stocks)
    adj_mat = (rand_matrix < density).float()
    adj_mat.fill_diagonal_(1.0)
    
    print(f"1. Generated Random Graph with {int(adj_mat.sum())} edges.")
    
    # 3. Generate Random Features
    # The content doesn't matter, just the shape [1, N, Hidden]
    embeddings = torch.randn(1, num_stocks, 64)
    
    # 4. Run GAT (Forward Pass)
    with torch.no_grad():
        # Input to GAT needs batch dim: [1, N, N]
        adj_batch = adj_mat.unsqueeze(0)
        
        # Get attention weights from the Industry GAT layer
        _, attn_weights = model.ind_gat(embeddings, adj_batch, require_weights=True)
        
        # Average over heads: [Batch, Heads, N, N] -> [N, N]
        avg_attn = attn_weights[0].mean(dim=0)

    # 5. The Leak Test (The "Right or Wrong" Check)
    # Logic: If Adj[i,j] == 0, then Attn[i,j] MUST be 0.
    
    # Identify the "Forbidden Zones" (White squares in the graph)
    forbidden_mask = (adj_mat == 0)
    
    # Sum up any attention found in forbidden zones
    leak_sum = avg_attn[forbidden_mask].sum().item()
    max_leak = avg_attn[forbidden_mask].max().item() if forbidden_mask.any() else 0.0
    
    print("\n--- 📊 Results ---")
    print(f"Total Attention Mass in Forbidden Zones: {leak_sum:.8f}")
    print(f"Max Single Leak Value: {max_leak:.8f}")
    
    if leak_sum < 1e-5:
        print("\n✅ PASS: Masking is perfect. No information leakage.")
    else:
        print("\n❌ FAIL: Graph is Leaky! The model is cheating.")
        print("   (Reason: You are likely using matrix multiplication instead of masking.)")

    # 6. Visualization
    plt.figure(figsize=(12, 6))
    
    # Input (The Rules)
    plt.subplot(1, 2, 1)
    sns.heatmap(adj_mat.numpy(), cmap="binary", cbar=False, linewidths=0.1, linecolor='gray')
    plt.title(f"Input: Random Graph Constraints\n(Black=Allowed, White=Forbidden)")
    plt.xlabel("Sender")
    plt.ylabel("Receiver")
    
    # Output (The Behavior)
    plt.subplot(1, 2, 2)
    sns.heatmap(avg_attn.numpy(), cmap="viridis", linewidths=0.1, linecolor='gray')
    plt.title("Output: Learned Attention\n(Should match the Black pattern exactly)")
    plt.xlabel("Sender")
    
    plt.tight_layout()
    plt.savefig("test/test_generic_gat_changed.png")
    print("\nVisual proof saved to: test/test_generic_gat_changed.png")

if __name__ == "__main__":
    # You can change density to test different graph types
    test_generic_masking(num_stocks=15, density=0.25)