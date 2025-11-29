import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import glob

def analyze_topology(market='custom'):
    print(f"--- Analyzing Graph Topology for Market: {market} ---")
    
    # 1. Find Correlation Files
    corr_dir = os.path.join("dataset", "corr", market)
    files = sorted(glob.glob(os.path.join(corr_dir, "*.csv")))
    
    if not files:
        print(f"❌ No correlation files found in {corr_dir}. Run build_dataset_yf.py first.")
        return

    print(f"Found {len(files)} monthly correlation matrices.")
    
    # Define linear range: 0.20 to 0.80 (inclusive) with 0.05 step
    thresholds = np.arange(0.20, 0.85, 0.05)
    
    # Storage for stats
    # Structure: {'0.20': {'density': [], ...}, ...}
    stats = {round(t, 2): {'density': [], 'islands': [], 'avg_degree': []} for t in thresholds}
    
    # Process every file (Month)
    for i, f in enumerate(files):
        # Progress indicator
        if i % 10 == 0: print(f"Processing month {i}/{len(files)}...", end='\r')
        
        df = pd.read_csv(f, index_col=0)
        corr = df.values
        np.fill_diagonal(corr, 0) # No self-loops
        abs_corr = np.abs(corr)
        
        num_nodes = corr.shape[0]
        max_edges = num_nodes * (num_nodes - 1)
        
        for t in thresholds:
            t_key = round(t, 2)
            
            # Create Adjacency Matrix
            adj = (abs_corr > t).astype(int)
            
            # 1. Density
            num_edges = adj.sum()
            density = num_edges / max_edges if max_edges > 0 else 0
            
            # 2. Fragmentation (Islands)
            graph = csr_matrix(adj)
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
            
            # 3. Avg Degree
            avg_degree = num_edges / num_nodes
            
            stats[t_key]['density'].append(density)
            stats[t_key]['islands'].append(n_components)
            stats[t_key]['avg_degree'].append(avg_degree)

    print("\nData processing complete. Generating plots...")

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Prepare data for plotting
    plot_data = []
    for t in thresholds:
        t_key = round(t, 2)
        plot_data.append({
            'Threshold': t_key,
            'Avg Density': np.mean(stats[t_key]['density']),
            'Avg Islands': np.mean(stats[t_key]['islands']),
            'Avg Degree': np.mean(stats[t_key]['avg_degree'])
        })
    df_plot = pd.DataFrame(plot_data)

    # Plot 1: Density
    sns.lineplot(data=df_plot, x='Threshold', y='Avg Density', ax=axes[0], marker='o', linewidth=2.5, color='tab:blue')
    axes[0].set_title("Graph Density vs Threshold")
    axes[0].set_ylabel("Density (0.0 - 1.0)")
    axes[0].grid(True)
    
    # Plot 2: Connectivity (Islands)
    sns.lineplot(data=df_plot, x='Threshold', y='Avg Islands', ax=axes[1], marker='o', linewidth=2.5, color='tab:red')
    axes[1].set_title("Fragmentation (Isolated Islands)")
    axes[1].set_ylabel("Count of Disconnected Components")
    # Horizontal line for "Ideal" (1 component)
    axes[1].axhline(1, color='green', linestyle='--', label='Fully Connected (1)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Degree
    sns.lineplot(data=df_plot, x='Threshold', y='Avg Degree', ax=axes[2], marker='o', linewidth=2.5, color='tab:green')
    axes[2].set_title("Average Neighbors per Stock")
    axes[2].set_ylabel("Node Degree")
    axes[2].grid(True)

    plt.suptitle(f"Graph Topology Sensitivity Analysis ({len(files)} Months Average)", fontsize=14)
    plt.tight_layout()
    
    out_path = "test/chart_topology_sensitivity_linear.png"
    plt.savefig(out_path)
    print(f"\n✅ Topology Linear Analysis Saved: {out_path}")
    
    # Print the table for reference
    print("\n--- Summary Data ---")
    print(df_plot.to_string(index=False))

if __name__ == "__main__":
    analyze_topology()