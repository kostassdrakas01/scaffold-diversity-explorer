from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Cluster import Butina
import numpy as np

def generate_fingerprints(mols):
    """Converts RDKit molecules into 2048-bit Morgan Fingerprints."""
    return [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

def calculate_ro5_properties(mol):
    """Calculates properties for Lipinski's Rule of 5."""
    if mol is None: return None
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol)
    }

def run_butina_clustering(fps, dist_cutoff=0.35):
    """
    Groups molecules based on Tanimoto Distance.
    Returns a tuple of clusters (each cluster is a tuple of indices).
    """
    dists = []
    nfps = len(fps)
    # Optimized distance matrix calculation for RDKit
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])
    
    return Butina.ClusterData(dists, nfps, dist_cutoff, isDistData=True)