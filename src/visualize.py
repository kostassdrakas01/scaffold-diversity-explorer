import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from rdkit import DataStructs

def get_np_fps(fps):
    """Converts RDKit fingerprints to a NumPy array for Scikit-Learn."""
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    return np.array(np_fps)

def plot_chemical_space(fps, diverse_indices, output_path='outputs/diversity_plot.png'):
    """
    Creates a t-SNE plot showing the full library vs. the selected subset.
    """
    print("Running t-SNE (this may take a minute)...")
    np_fps = get_np_fps(fps)
    
    # Reduce 2048-bit fingerprints to 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(np_fps)
    
    plt.figure(figsize=(12, 8))
    
    # Plot the background (all compounds)
    plt.scatter(coords[:, 0], coords[:, 1], c='lightgrey', s=10, alpha=0.5, label='Library Compounds')
    
    # Plot the selected centroids
    plt.scatter(coords[diverse_indices, 0], coords[diverse_indices, 1], 
                c='red', s=30, edgecolors='black', label='Diverse Subset (Centroids)')
    
    plt.title('Chemical Space Coverage: Diversity Selection', fontsize=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()