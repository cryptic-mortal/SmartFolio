import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from sklearn.decomposition import PCA

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import TemporalHGAT

def setup_dummy_model(num_stocks=12, input_dim=6, lookback=30, hidden_dim=64):
    torch.manual_seed(123)
    model = TemporalHGAT(
        num_stocks=num_stocks,
        input_dim=input_dim,
        lookback=lookback,
        hidden_dim=hidden_dim,
        num_heads=4
    )
    model.eval()
    return model

def generate_trend_data(num_stocks, lookback, input_dim):
    # Same generator: 3 Groups (Up, Down, Noise)
    ts_data = torch.randn(num_stocks, lookback, input_dim) * 0.1 
    group_size = num_stocks // 3
    
    # Uptrend
    for i in range(0, group_size):
        ts_data[i, :, 0] = torch.linspace(0, 2.0, lookback)
    # Downtrend
    for i in range(group_size, 2*group_size):
        ts_data[i, :, 0] = torch.linspace(0, -2.0, lookback)
    
    return ts_data

def visualize_dynamics():
    print("--- 🎬 Visualizing LSTM Thinking Dynamics (Time Evolution) ---")
    
    N = 12
    L = 30
    D = 6
    Hidden = 64
    
    model = setup_dummy_model(N, D, L, hidden_dim=Hidden)
    ts_data = generate_trend_data(N, L, D)
    
    # 1. Run LSTM and keep ALL time steps
    # Input: [N, L, D]
    with torch.no_grad():
        # output: [N, L, Hidden]
        lstm_out, _ = model.lstm(ts_data)
        
    print(f"LSTM Output Shape: {lstm_out.shape} (Stocks, Days, Neurons)")
    
    # 2. PCA Projection (Reduce 64D -> 2D for plotting)
    # We flatten [N, L, Hidden] -> [N*L, Hidden] to fit PCA on all states
    flat_states = lstm_out.reshape(-1, Hidden).numpy()
    
    pca = PCA(n_components=2)
    pca_states = pca.fit_transform(flat_states)
    
    # Reshape back to [N, L, 2]
    trajectories = pca_states.reshape(N, L, 2)
    
    # 3. Plot
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Colors for groups
    colors = ['green', 'red', 'gray']
    labels = ['Uptrend', 'Downtrend', 'Noise']
    
    for i in range(N):
        # Determine group
        group_idx = i // 4
        color = colors[group_idx]
        label = labels[group_idx] if i % 4 == 0 else None
        
        # Get path for this stock
        path = trajectories[i] # [30, 2]
        
        # Plot line
        plt.plot(path[:, 0], path[:, 1], color=color, alpha=0.6, linewidth=1.5, label=label)
        
        # Mark Start (Day 0) and End (Day 30)
        plt.scatter(path[0, 0], path[0, 1], color=color, marker='o', s=30, alpha=0.5) # Start
        plt.scatter(path[-1, 0], path[-1, 1], color=color, marker='>', s=100, edgecolors='black') # End arrow
        
        # Annotate End
        if i % 4 == 0: # Label first of each group
            plt.text(path[-1, 0], path[-1, 1], f" {labels[group_idx]}", fontsize=12, fontweight='bold', color=color)

    plt.title(f"LSTM 'Thought Process' Over 30 Days (PCA Projection)\nSee how different trends drift apart over time", fontsize=14)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()
    
    save_path = "test/brain_1_lstm_dynamics.png"
    plt.savefig(save_path)
    print(f"✅ Dynamics Chart Saved: {save_path}")

if __name__ == "__main__":
    visualize_dynamics()