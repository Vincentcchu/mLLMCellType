#!/usr/bin/env python3
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_LIB = PROJECT_ROOT / "mLLMCelltype" / "python"
if str(LOCAL_LIB) not in sys.path:
    sys.path.insert(0, str(LOCAL_LIB))

from mllmcelltype import interactive_consensus_annotation


def _load_openrouter_key(config_path: str = "api_keys.json") -> str:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["openrouter_api_key"]


def main():
    input_path = Path("/tmp/input.h5ad")
    tissue_name = "brain"
    output_path = Path("out.h5ad")
    summary_path = Path("summary.json")
    run_id = "brain_debug"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    wall_start = datetime.now()
    process_start = time.perf_counter()
    print(f"Process start: {wall_start.isoformat(timespec='seconds')}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    adata = sc.read_h5ad(input_path)

    if "leiden" not in adata.obs.columns:
        if "log1p" not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        if "X_pca" not in adata.obsm:
            sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
            sc.pp.pca(adata, use_highly_variable=True)
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
        sc.tl.leiden(adata, resolution=0.8)

    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
    marker_genes = {}
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
    ]

    annotation_start = time.perf_counter()
    try:
        consensus_results = interactive_consensus_annotation(
            marker_genes=marker_genes,
            species="human",
            tissue=tissue_name,
            models=models,
            api_keys={"openrouter": api_key},
            consensus_threshold=0.7,
            max_discussion_rounds=2,
            track_usage=True,
        )
    except Exception as exc:  # pragma: no cover - model/service failures
        annotation_end = time.perf_counter()
        summary = {
            "run_id": run_id,
            "process_start": wall_start.isoformat(timespec="seconds"),
            "process_end": datetime.now().isoformat(timespec="seconds"),
            "total_duration_seconds": round(time.perf_counter() - process_start, 2),
            "annotation_phase_seconds": round(annotation_end - annotation_start, 2),
            "annotation_model": "openrouter consensus (mixed free models)",
            "tokens_used": None,
            "credits_used": None,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "tissue": tissue_name,
            "status": "error",
            "error": str(exc),
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Error during consensus call: {exc}")
        sys.exit(1)

    annotation_end = time.perf_counter()

    if not consensus_results or "consensus" not in consensus_results:
        summary = {
            "run_id": run_id,
            "process_start": wall_start.isoformat(timespec="seconds"),
            "process_end": datetime.now().isoformat(timespec="seconds"),
            "total_duration_seconds": round(time.perf_counter() - process_start, 2),
            "annotation_phase_seconds": round(annotation_end - annotation_start, 2),
            "annotation_model": "openrouter consensus (mixed free models)",
            "tokens_used": None,
            "credits_used": None,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "tissue": tissue_name,
            "status": "no_consensus",
            "error": "No consensus returned; see logs for details",
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print("No consensus returned; exiting")
        sys.exit(1)

    annotations = consensus_results["consensus"]
    adata.obs["openrouter_cell_type"] = adata.obs["leiden"].astype(str).map(annotations)

    wall_end = datetime.now()
    process_end = time.perf_counter()
    usage_info = consensus_results.get("usage", {})
    total_tokens = usage_info.get("total_tokens")
    credits_used = usage_info.get("total_credits")

    summary = {
        "run_id": run_id,
        "process_start": wall_start.isoformat(timespec="seconds"),
        "process_end": wall_end.isoformat(timespec="seconds"),
        "total_duration_seconds": round(process_end - process_start, 2),
        "annotation_phase_seconds": round(annotation_end - annotation_start, 2),
        "annotation_model": "openrouter consensus (mixed free models)",
        "tokens_used": total_tokens,
        "credits_used": credits_used,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tissue": tissue_name,
        "status": "completed",
    }

    adata.write(output_path)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nRun summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
