import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Import the specialized functions
from src.utils import (
    preprocess_and_clean, 
    generate_fingerprints, 
    run_butina_clustering, 
    calculate_ro5_metrics,
    get_internal_diversity
)
from src.visualize import plot_property_distributions, export_cluster_grid

# --- 1. Configuration ---
DATA_PATH = 'data/library.csv'
OUTPUT_DIR = 'outputs'
BUDGET = 500  
DIST_THRESHOLD = 0.35 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # --- 2. Data Acquisition and Chemical Cleaning ---
    print("Initializing Chemoinformatics Pipeline...")
    df = pd.read_csv(DATA_PATH)
    
    df['ROMol'] = preprocess_and_clean(df['smiles'])
    df = df.dropna(subset=['ROMol']).reset_index(drop=True)
    print(f"Standardization complete. {len(df)} valid molecules retained.")

    # --- 3. Physicochemical Profiling ---
    print("Calculating Lipinski Rule of 5 parameters...")
    ro5_data = df['ROMol'].apply(calculate_ro5_metrics)
    ro5_df = pd.DataFrame(list(ro5_data))
    df = pd.concat([df, ro5_df], axis=1)

    # --- 4. Molecular Vectorization and Clustering ---
    print("Generating 2048-bit Morgan Fingerprints (ECFP4)...")
    fps = generate_fingerprints(df['ROMol'])
    
    print(f"Executing Butina Clustering (Cutoff: {DIST_THRESHOLD})...")
    clusters = run_butina_clustering(fps, dist_cutoff=DIST_THRESHOLD)
    print(f"Identified {len(clusters)} distinct chemical clusters.")

    # --- 5. Selection and Validation ---
    diverse_indices = [cluster[0] for cluster in clusters[:BUDGET]]
    diverse_df = df.iloc[diverse_indices].copy()
    diverse_fps = [fps[i] for i in diverse_indices]
    
    div_score = get_internal_diversity(diverse_fps)
    print(f"Validation Metric: Mean Internal Tanimoto Distance = {div_score:.3f}")

    # --- 6. Low-Dimensional Projection (t-SNE) ---
    print("Generating Chemical Space Visualization via t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    np_fps = np.array([list(fp) for fp in fps])
    coords = tsne.fit_transform(np_fps)
    
    df['tsne_1'] = coords[:, 0]
    df['tsne_2'] = coords[:, 1]

    plt.figure(figsize=(12, 8))
    plt.scatter(df['tsne_1'], df['tsne_2'], c='lightgrey', s=5, label='Total Library', alpha=0.4)
    plt.scatter(df.iloc[diverse_indices]['tsne_1'], df.iloc[diverse_indices]['tsne_2'], 
                c='crimson', s=20, label='Diverse Subset', edgecolors='black', linewidth=0.5)
    
    plt.title(f"Chemical Space Distribution (Mean Internal Distance: {div_score:.2f})")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/chemical_space_map.png", dpi=300)
    plt.close()

    # --- 7. Supplementary Visualizations ---
    print("Generating property distributions and structure grids...")
    plot_property_distributions(df, f"{OUTPUT_DIR}/property_distributions.png")
    
    # We pass the diverse_df to export a grid of the selected molecules
    export_cluster_grid(diverse_df, f"{OUTPUT_DIR}/diverse_scaffold_examples.png")

    # --- 8. Exporting Results ---
    diverse_df.to_csv(f"{OUTPUT_DIR}/diverse_subset_results.csv", index=False)
    print(f"Pipeline execution successful. Results exported to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
