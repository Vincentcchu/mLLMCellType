"""Tracked example for mLLMCelltype using OpenRouter with token/credit metrics."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import scanpy as sc

# Ensure we import the local library (not a different installed version)
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_LIB = PROJECT_ROOT / "mLLMCelltype" / "python"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from mllmcelltype import annotate_clusters, interactive_consensus_annotation

def _load_openrouter_key(config_path: str = "api_keys.json") -> str:
    """Load OpenRouter API key from a simple JSON file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["openrouter_api_key"]


def _summarize_usage(usage: dict | None) -> tuple[int | None, float | None]:
    """Extract total tokens and credits from a usage payload."""
    if not usage:
        return None, None

    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    credits_used = usage.get("total_cost") or usage.get("cost")

    return total_tokens, credits_used


def main():
    wall_start = datetime.now()
    process_start = time.perf_counter()
    print(f"Process start: {wall_start.isoformat(timespec='seconds')}")

    # Load AnnData
    input_path = "/cs/student/projects2/aisd/2024/shekchu/projects/data/breast/h5ad_l3/Data_Bassez2021_Breast_l3.h5ad"
    adata = sc.read_h5ad(input_path)

    # Basic preprocessing and clustering only if missing
    if "leiden" not in adata.obs.columns:
        if "log1p" not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        if "X_pca" not in adata.obsm:
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            sc.pp.pca(adata, use_highly_variable=True)
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
        sc.tl.leiden(adata, resolution=0.8)

    # Differential expression for marker genes
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
    marker_genes: dict[str, list[str]] = {}
    for i in range(len(adata.obs["leiden"].cat.categories)):
        genes = [adata.uns["rank_genes_groups"]["names"][str(i)][j] for j in range(10)]
        marker_genes[str(i)] = genes

    api_key = _load_openrouter_key()
    os.environ["OPENROUTER_API_KEY"] = api_key

    models = [
        {"provider": "openrouter", "model": "deepseek/deepseek-r1-0528:free"},
        {"provider": "openrouter", "model": "moonshotai/kimi-k2:free"},
        {"provider": "openrouter", "model": "google/gemini-2.0-flash-exp:free"},
        {"provider": "openrouter", "model": "qwen/qwq-32b:free"},
        # {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.5"},   
        # {"provider": "openrouter", "model": "google/gemini-3-pro-preview"},   
        # {"provider": "openrouter", "model": "moonshotai/kimi-k2.5"},
        # {"provider": "openrouter", "model": "openai/gpt-5.2"}
    ]

    annotation_start = time.perf_counter()
    consensus_results = interactive_consensus_annotation(
        marker_genes=marker_genes,
        species="human",
        tissue="colorectal",
        models=models,
        api_keys={"openrouter": api_key},
        consensus_threshold=0.7,
        max_discussion_rounds=2,
        track_usage=True,
    )
    annotation_end = time.perf_counter()

    annotations = consensus_results["consensus"]

    # Attach annotations
    adata.obs["openrouter_cell_type"] = adata.obs["leiden"].astype(str).map(annotations)

    # Calculate timings and usage
    wall_end = datetime.now()
    process_end = time.perf_counter()
    usage_info = consensus_results.get("usage", {})
    total_tokens = usage_info.get("total_tokens")
    credits_used = usage_info.get("total_credits")

    summary = {
        "process_start": wall_start.isoformat(timespec="seconds"),
        "process_end": wall_end.isoformat(timespec="seconds"),
        "total_duration_seconds": round(process_end - process_start, 2),
        "annotation_phase_seconds": round(annotation_end - annotation_start, 2),
        "annotation_model": "openrouter consensus (mixed free models)",
        "tokens_used": total_tokens,
        "credits_used": credits_used,
    }

    print("\nRun summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    # Persist annotated data using tissue_dataset_task pattern
    tissue_name = "breast"
    dataset_stem = Path(input_path).stem
    dataset_clean = dataset_stem.replace("Data_", "")
    tissue_suffix = f"_{tissue_name.capitalize()}"
    if dataset_clean.endswith(tissue_suffix):
        dataset_clean = dataset_clean[: -len(tissue_suffix)]
    dataset_clean = dataset_clean.replace("_", "").lower()

    output_filename = f"{tissue_name}_{dataset_clean}_malignant_classification.h5ad"
    output_path = output_filename

    adata.write(output_path)
    print(f"\nAnnotated data saved to {output_path}")


if __name__ == "__main__":
    main()
