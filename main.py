import pandas as pd
from rdkit.Chem import PandasTools
from src.utils import generate_fingerprints, run_butina_clustering
from src.visualize import plot_chemical_space

# 1. Configuration
DATA_PATH = 'data/library.csv' # Ensure you have a SMILES column named 'smiles'
BUDGET = 500
DIST_CUTOFF = 0.35 # Tanimoto Distance threshold

# 2. Load and Clean
print("Loading data...")
df = pd.read_csv(DATA_PATH)
PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'ROMol')
df = df.dropna(subset=['ROMol']).reset_index(drop=True)

# 3. Generate Fingerprints & Cluster
print("Generating fingerprints...")
fps = generate_fingerprints(df['ROMol'])

print("Clustering molecules...")
clusters = run_butina_clustering(fps, dist_cutoff=DIST_CUTOFF)

# 4. Diversified Selection
# Sort by cluster size (pick centroids of the most representative families)
sorted_clusters = sorted(clusters, key=len, reverse=True)
diverse_indices = [c[0] for c in sorted_clusters[:BUDGET]]

# 5. Export Results
diverse_df = df.iloc[diverse_indices].copy()
diverse_df.to_csv('outputs/diverse_subset.csv', index=False)
print(f"Success! Selected {len(diverse_indices)} molecules.")

# 6. Visualize
plot_chemical_space(fps, diverse_indices)