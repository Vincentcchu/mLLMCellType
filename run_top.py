# Example of using mLLMCelltype for single-cell RNA-seq cell type annotation with Scanpy
# Configured to use only GPT-4o
import scanpy as sc
import pandas as pd
from mllmcelltype import annotate_clusters, interactive_consensus_annotation
import os

# Note: Logging is automatically configured when importing mllmcelltype
# You can customize logging if needed using the logging module

# Load your single-cell RNA-seq dataset in AnnData format
adata = sc.read_h5ad('../../data/dataset_restricted.h5ad')  # Replace with your scRNA-seq dataset path

# Perform Leiden clustering for cell population identification if not already done
if 'leiden' not in adata.obs.columns:
    print("Computing leiden clustering for cell population identification...")
    # Preprocess single-cell data: normalize counts and log-transform for gene expression analysis
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize to 10,000 counts per cell
        sc.pp.log1p(adata)  # Log-transform normalized counts

    # Dimensionality reduction: calculate PCA for scRNA-seq data
    if 'X_pca' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)  # Select informative genes
        sc.pp.pca(adata, use_highly_variable=True)  # Compute principal components

    # Cell clustering: compute neighborhood graph and perform Leiden community detection
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)  # Build KNN graph for clustering
    sc.tl.leiden(adata, resolution=0.8)  # Identify cell populations using Leiden algorithm
    print(f"Leiden clustering completed, identified {len(adata.obs['leiden'].cat.categories)} distinct cell populations")

# Identify marker genes for each cell cluster using differential expression analysis
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')  # Wilcoxon rank-sum test for marker detection

# Extract top marker genes for each cell cluster to use in cell type annotation
marker_genes = {}
for i in range(len(adata.obs['leiden'].cat.categories)):
    # Select top 10 differentially expressed genes as markers for each cluster
    genes = [adata.uns['rank_genes_groups']['names'][str(i)][j] for j in range(10)]
    marker_genes[str(i)] = genes

# IMPORTANT: mLLMCelltype requires gene symbols (e.g., KCNJ8, PDGFRA) not Ensembl IDs (e.g., ENSG00000176771)
# If your AnnData object uses Ensembl IDs, convert them to gene symbols for accurate annotation:
# Example conversion code:
# if 'Gene' in adata.var.columns:  # Check if gene symbols are available in the metadata
#     gene_name_dict = dict(zip(adata.var_names, adata.var['Gene']))
#     marker_genes = {cluster: [gene_name_dict.get(gene_id, gene_id) for gene_id in genes]
#                    for cluster, genes in marker_genes.items()}

# IMPORTANT: mLLMCelltype requires numeric cluster IDs
# The 'cluster' column must contain numeric values or values that can be converted to numeric.
# Non-numeric cluster IDs (e.g., "cluster_1", "T_cells", "7_0") may cause errors or unexpected behavior.
# If your data contains non-numeric cluster IDs, create a mapping between original IDs and numeric IDs:
# Example standardization code:
# original_ids = list(marker_genes.keys())
# id_mapping = {original: idx for idx, original in enumerate(original_ids)}
# marker_genes = {str(id_mapping[cluster]): genes for cluster, genes in marker_genes.items()}

# Configure API key for GPT-4o from JSON file
import json

# Load API key from external JSON file
try:
    with open('api_keys.json', 'r') as f:
        api_key_data = json.load(f)
    os.environ["OPENAI_API_KEY"] = api_key_data["openai_api_key"]
except FileNotFoundError:
    print("Error: api_keys.json file not found. Please create this file with your OpenAI API key.")
    exit(1)
except (KeyError, json.JSONDecodeError) as e:
    print(f"Error reading API key from JSON file: {e}")
    print("Make sure api_keys.json contains a valid JSON with 'openai_api_key' and 'openrouter_api_key' fields.")
    exit(1)

# Set OpenRouter API key from the same JSON file
try:
    os.environ["OPENROUTER_API_KEY"] = api_key_data["openrouter_api_key"]
except KeyError:
    print("Warning: 'openrouter_api_key' not found in api_keys.json. OpenRouter models may not work correctly.")


top_models_results = interactive_consensus_annotation(
    marker_genes=marker_genes,
    species="human",
    tissue="blood",
    models=[
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},   
        {"provider": "openrouter", "model": "google/gemini-2.5-pro"},   
        {"provider": "openrouter", "model": "qwen/qwen3-235b-a22b-2507:free"},
        {"provider": "openrouter", "model": "openai/o4-mini"}
    ],
    consensus_threshold=0.7,
    max_discussion_rounds=2
)

# Retrieve final cell type annotations from GPT-4o
final_annotations = top_models_results["consensus"]

# Integrate cell type annotations into the original AnnData object
adata.obs['gpt4o_cell_type'] = adata.obs['leiden'].astype(str).map(final_annotations)

# Add confidence metrics (though with single model, these will be simpler)
adata.obs['annotation_confidence'] = adata.obs['leiden'].astype(str).map(top_models_results["consensus_proportion"])

# Prepare for visualization: compute UMAP embeddings if not already available
# UMAP provides a 2D representation of cell populations for visualization
if 'X_umap' not in adata.obsm:
    print("Computing UMAP coordinates...")
    # Make sure neighbors are computed first
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)
    print("UMAP coordinates computed")

# Visualize results
import matplotlib.pyplot as plt

# Set figure size and style
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

# Create UMAP visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sc.pl.umap(adata, color='gpt4o_cell_type', legend_loc='on data',
         frameon=True, title='GPT-4o Cell Type Annotations',
         palette='tab20', size=50, legend_fontsize=12,
         legend_fontoutline=2, ax=ax)

# Optional: visualize confidence scores
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sc.pl.umap(adata, color='annotation_confidence', ax=ax, title='Annotation Confidence',
         cmap='viridis', vmin=0, vmax=1, size=30)

plt.tight_layout()
plt.show()

# Print summary of annotations
print("\nCell Type Annotation Summary:")
print(f"Total clusters: {len(final_annotations)}")
print("\nCluster -> Cell Type mapping:")
for cluster, cell_type in final_annotations.items():
    cluster_size = sum(adata.obs['leiden'] == cluster)
    print(f"Cluster {cluster}: {cell_type} ({cluster_size} cells)")

# Save results
adata.write('mllm_top_models.h5ad')
print("\nAnnotated data saved to 'mllm_top_models.h5ad'")