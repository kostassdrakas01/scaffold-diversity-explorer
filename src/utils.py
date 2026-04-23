import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.ML.Cluster import Butina

def preprocess_and_clean(smiles_list):
    """
    Standardizes molecules: Strips salts, keeps largest fragment, 
    and canonicalizes SMILES for consistent clustering.
    """
    remover = SaltRemover()
    cleaned_mols = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Strip salts and neutralize
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            # Standardize tautomers/canonicalization happens here
            cleaned_mols.append(mol)
        else:
            cleaned_mols.append(None)
            
    return cleaned_mols

def generate_fingerprints(mols):
    """
    ECFP4 (Morgan radius 2) with 2048-bit resolution. 
    Explicitly defined for reproducibility in diversity selection.
    """
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]

def calculate_ro5_metrics(mol):
    """
    Calculates Lipinski's Rule of 5 properties to ensure 
    selected diverse subsets are also 'Drug-Like'.
    """
    if mol is None: return None
    return {
        'MW': round(Descriptors.MolWt(mol), 2),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RO5_Pass': (Descriptors.MolWt(mol) <= 500 and 
                     Descriptors.MolLogP(mol) <= 5 and 
                     Descriptors.NumHDonors(mol) <= 5 and 
                     Descriptors.NumHAcceptors(mol) <= 10)
    }

def run_butina_clustering(fps, dist_cutoff=0.35):
    """
    Groups molecules by Tanimoto Distance. 
    0.35 cutoff is used as the standard scaffold-diversity threshold.
    """
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])
    
    return Butina.ClusterData(dists, nfps, dist_cutoff, isDistData=True)

def get_internal_diversity(fps):
    """
    Calculates the Mean Internal Tanimoto Distance. 
    Validation metric: Higher distance = more successful diversity selection.
    """
    if len(fps) < 2: return 0
    dists = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1-x for x in sims])
    return np.mean(dists)
