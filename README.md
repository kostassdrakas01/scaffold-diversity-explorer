Scaffold Diversity Explorer: Optimizing Chemical Space Coverage
Project Overview

In early-stage drug discovery, high-throughput screening (HTS) is expensive. Testing 10,000+ compounds is often cost-prohibitive for smaller labs or specific assays.

The Solution: This project implements an Unsupervised Machine Learning pipeline to select a representative, chemically diverse subset of compounds. By using Butina Clustering, we identify structural "islands" and select the most representative member (centroid) of each family, maximizing the chemical space explored while minimizing the laboratory budget.
Key Features

    Vectorization: Converts SMILES into 2048-bit Morgan Fingerprints (ECFP4 equivalent).

    Clustering: Industry-standard Butina Clustering using Tanimoto Similarity.

    Dimensionality Reduction: Implements t-SNE to project high-dimensional chemical space into a 2D map.

    Automated Selection: Automatically picks the top N diverse centroids based on a user-defined budget.

Visualizing Chemical Space

The plot below demonstrates the pipeline's effectiveness. The gray background represents the full Tox21 library (~8k molecules), while the red markers indicate the diverse subset selected by the algorithm.

Note: Observe how the red points "anchor" the various clusters, ensuring no major chemical scaffold is left untested.
Technical Stack

    RDKit: Molecular sanitization and fingerprinting.

    Scikit-Learn: t-SNE dimensionality reduction.

    Pandas: Data orchestration.

    Matplotlib: High-fidelity visualization.

Project Structure
Plaintext

scaffold_explorer/
├── src/
│   ├── utils.py        # Core RDKit logic (Fingerprints, Butina)
│   └── visualize.py    # t-SNE and plotting functions
├── data/               # Input CSVs (e.g., library.csv)
├── outputs/            # Generated diverse CSVs and plots
├── main.py             # Entry point script
└── requirements.txt    # Project dependencies

Installation & Usage

    Clone the repo:
    Bash

    git clone https://github.com/YOUR_USERNAME/scaffold-diversity-explorer.git
    cd scaffold_diversity_explorer

    Set up Virtual Environment:
    Bash

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    Run the pipeline:
    Place your SMILES data in data/library.csv and run:
    Bash

    python3 main.py

 Impact Statement

By using this diversity-selection approach, laboratories can reduce their screening workload by up to 90% while maintaining a high probability of discovering hits across multiple chemical classes. This directly reduces reagent costs, plate usage, and equipment time.

Author: Konstantinos Sdrakas

Contact: kostaskostassdrakas@gmail.com
LinkedIn: www.linkedin.com/in/kostas-sdrakas

Topic: Chemoinformatics & Computational Drug Discovery
