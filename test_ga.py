"""
Test script to demonstrate GA stock selection with verbose output
"""
import sys
import numpy as np
import pickle
from gen_data.generate_expert_ga import generate_expert_strategy_ga

# Load a sample from custom dataset
data_dir = 'dataset_default/data_train_predict_custom/1_hy/'
import os
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])

# Find a file with data
sample_file = None
for fname in files[20:40]:
    with open(os.path.join(data_dir, fname), 'rb') as f:
        data = pickle.load(f)
    if data['features'].numel() > 0:
        sample_file = fname
        break

if sample_file is None:
    print("No valid data file found!")
    sys.exit(1)

print(f"Testing GA with data from: {sample_file}")
print(f"{'='*70}\n")

# Extract data
features = data['features'].numpy()
correlation_matrix = data['corr'].numpy()
industry_matrix = data['industry_matrix'].numpy()
returns = data['labels'].numpy()

print(f"Stock universe: {len(returns)} stocks")
print(f"Returns: {returns}")
print(f"Mean return: {np.mean(returns):.4f}")
print(f"Std return: {np.std(returns):.4f}")
print()

# Test with different risk categories
for risk_cat in ['conservative', 'moderate', 'aggressive']:
    print(f"\n{'#'*70}")
    print(f"# TESTING {risk_cat.upper()} STRATEGY")
    print(f"{'#'*70}")
    
    selected = generate_expert_strategy_ga(
        returns=returns,
        industry_relation_matrix=industry_matrix,
        correlation_matrix=correlation_matrix,
        risk_category=risk_cat,
        ga_generations=30,
        verbose=True
    )
    
    print(f"\n{'='*70}\n")
