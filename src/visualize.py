import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import Draw
import os

def plot_property_distributions(df, output_path):
    """
    Creates a grid of histograms to compare the properties 
    of the diverse subset vs. the total library.
    """
    properties = ['MW', 'LogP', 'HBD', 'HBA']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, prop in enumerate(properties):
        sns.histplot(df[prop], ax=axes[i], kde=True, color='blue', label='Full Library')
        plt.legend()
        axes[i].set_title(f'Distribution of {prop}')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def export_cluster_grid(diverse_df, output_path, n_mols=12):
    """
    Generates an image grid of the top diverse molecules.
    This provides a visual qualitative check of the scaffolds.
    """
    mols = diverse_df['ROMol'].tolist()[:n_mols]
    legends = [f"ID: {id}" for id in diverse_df['mol_id'].tolist()[:n_mols]]
    
    img = Draw.MolsToGridImage(
        mols, 
        molsPerRow=4, 
        subImgSize=(300, 300), 
        legends=legends
    )
    img.save(output_path)
