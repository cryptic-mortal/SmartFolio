import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

# Add root to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import TemporalHGAT

def setup_dummy_model(num_stocks=12, input_dim=6, lookback=30, hidden_dim=64):
    """Initializes the model with random weights."""
    torch.manual_seed(42)
    model = TemporalHGAT(
        num_stocks=num_stocks,
        input_dim=input_dim,
        lookback=lookback,
        hidden_dim=hidden_dim,
        num_heads=4,
        no_ind=False,
        no_neg=False
    )
    model.eval()
    return model

def generate_trend_data(num_stocks, lookback, input_dim):
    """
    Generates time-series data with 3 distinct patterns:
    - Group 1 (Stocks 0-3): STRONG UPTREND
    - Group 2 (Stocks 4-7): STRONG DOWNTREND
    - Group 3 (Stocks 8-11): RANDOM NOISE
    """
    # Initialize with low noise
    ts_data = torch.randn(num_stocks, lookback, input_dim) * 0.1 
    
    group_size = num_stocks // 3
    
    # Group 1: Uptrend (Price feature 0 goes 0.0 -> 2.0)
    for i in range(0, group_size):
        ts_data[i, :, 0] = torch.linspace(0, 2.0, lookback)
    
    # Group 2: Downtrend (Price feature 0 goes 0.0 -> -2.0)
    for i in range(group_size, 2*group_size):
        ts_data[i, :, 0] = torch.linspace(0, -2.0, lookback)
        
    # Group 3: Random Noise (Keeps the random init)
    
    return ts_data

def test_lstm_brain():
    print("--- 🧠 Testing Brain Component 1: LSTM (Temporal Encoder) ---")
    
    # Settings
    N = 12  # 12 Stocks (4 Up, 4 Down, 4 Noise)
    L = 30  # Lookback
    D = 6   # Features
    
    # 1. Setup
    model = setup_dummy_model(N, D, L)
    ts_data = generate_trend_data(N, L, D)
    
    # 2. Simulate Input Shaping
    # In the real model, this data comes flattened.
    # We verify the model can ingest the raw [N, L, D] format correctly.
    # Input shape required by LSTM: [Batch, Length, Features]
    # We treat the stocks as a batch.
    lstm_input = ts_data.view(N, L, D)
    
    print(f"1. Input Data Shape: {lstm_input.shape}")
    print("   -> Simulating 3 Market Regimes (Up, Down, Noise)")

    # 3. Run The Brain (LSTM Pass)
    with torch.no_grad():
        # output: [Batch, Length, Hidden]
        lstm_out, _ = model.lstm(lstm_input)
        
        # We take the LAST hidden state as the "Embedding"
        # shape: [N, Hidden]
        node_embeddings = lstm_out[:, -1, :]
    
    print(f"2. Output Embedding Shape: {node_embeddings.shape}")
    print("   -> Compressing 30 days of history into 64 distinct features.")

    # 4. Visualization
    print("3. Generating Diagnostic Chart...")
    plt.figure(figsize=(12, 8))
    
    # Create Heatmap
    sns.heatmap(node_embeddings.detach().numpy(), cmap="coolwarm", center=0)
    
    # Add Dividers
    plt.axhline(4, color='white', linewidth=2)
    plt.axhline(8, color='white', linewidth=2)
    
    # Labels
    plt.title("LSTM Memory Test: Do different trends look different?\n(If this map is uniform color, the LSTM is broken)", fontsize=14)
    plt.ylabel("Stock Index")
    plt.xlabel("Hidden Features (Neurons)")
    
    # Custom Y-Ticks to show groups
    plt.yticks([2, 6, 10], ['UPTREND\n(Stocks 0-3)', 'DOWNTREND\n(Stocks 4-7)', 'NOISE\n(Stocks 8-11)'], rotation=0)
    
    plt.tight_layout()
    plt.savefig("test/brain_1_lstm_only.png")
    print("✅ Test Complete. Results saved to: test/brain_1_lstm_only.png")

if __name__ == "__main__":
    test_lstm_brain()