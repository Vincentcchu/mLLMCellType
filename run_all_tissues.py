#!/usr/bin/env python3
"""
Automate running mLLMCelltype tracked annotator on h5ad files across tissues.
Creates a temporary runner per dataset based on run_tracked without modifying it.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from string import Template
from typing import List, Tuple

DATA_BASE_DIR = "/cs/student/projects2/aisd/2024/shekchu/projects/data"
DEFAULT_H5AD_SUBDIR = "h5ad_unlabelled_clustered"
DEFAULT_OUTPUT_DIR = "output"

RUNNER_TEMPLATE = Template(
    """#!/usr/bin/env python3
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
    input_path = Path("$input_path")
    tissue_name = "$tissue"
    output_path = Path("$output_path")
    summary_path = Path("$summary_path")
    run_id = "$run_id"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    wall_start = datetime.now()
    process_start = time.perf_counter()
    print(f"Process start: {wall_start.isoformat(timespec='seconds')}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    adata = sc.read_h5ad(input_path)

    if "cluster" not in adata.obs.columns:
        raise ValueError(
            f"Error: 'cluster' column not found in {input_path}. "
            "Dataset must contain pre-computed clustering information in adata.obs['cluster']."
        )

    # Ensure cluster column is categorical with string categories
    if not hasattr(adata.obs["cluster"], "cat"):
        adata.obs["cluster"] = adata.obs["cluster"].astype("category")
    else:
        # Convert categories to strings if they're not already
        adata.obs["cluster"] = adata.obs["cluster"].cat.rename_categories(
            str
        )

    # Check cluster sizes
    cluster_counts = adata.obs["cluster"].value_counts()
    print(f"\\nCluster sizes: {dict(cluster_counts)}")
    
    # Separate clusters by size for different processing
    single_cell_clusters = cluster_counts[cluster_counts == 1].index.tolist()
    multi_cell_clusters = cluster_counts[cluster_counts > 1].index.tolist()
    
    marker_genes = {}
    
    # Process multi-cell clusters with Wilcoxon test for differential expression
    if multi_cell_clusters:
        adata_multi = adata[adata.obs["cluster"].isin(multi_cell_clusters)].copy()
        adata_multi.obs["cluster"] = adata_multi.obs["cluster"].cat.remove_unused_categories()
        
        sc.tl.rank_genes_groups(adata_multi, "cluster", method="wilcoxon")
        for cluster_id in adata_multi.obs["cluster"].cat.categories:
            genes = [adata_multi.uns["rank_genes_groups"]["names"][cluster_id][j] for j in range(10)]
            marker_genes[cluster_id] = genes
    
    # Process single-cell clusters using top expressed genes
    if single_cell_clusters:
        print(f"Processing {len(single_cell_clusters)} single-cell cluster(s) using top expressed genes: {single_cell_clusters}")
        for cluster_id in single_cell_clusters:
            cell_data = adata[adata.obs["cluster"] == cluster_id]
            # Get top 10 expressed genes for this single cell
            gene_expression = cell_data.X.toarray().flatten() if hasattr(cell_data.X, 'toarray') else cell_data.X.flatten()
            top_gene_indices = gene_expression.argsort()[-10:][::-1]
            top_genes = cell_data.var_names[top_gene_indices].tolist()
            marker_genes[cluster_id] = top_genes

    api_key = _load_openrouter_key()
    os.environ["OPENROUTER_API_KEY"] = api_key

    models = [
        {"provider": "openrouter", "model": "openai/gpt-5.1"},
        {"provider": "openrouter", "model": "google/gemini-3.1-pro-preview"},  
        {"provider": "openrouter", "model": "qwen/qwen3.5-plus-02-15"},    
    ]

    annotation_start = time.perf_counter()
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
    annotation_end = time.perf_counter()

    annotations = consensus_results["consensus"]
    adata.obs["openrouter_cell_type"] = adata.obs["cluster"].astype(str).map(annotations)

    wall_end = datetime.now()
    process_end = time.perf_counter()
    usage_info = consensus_results.get("usage", {})
    total_tokens = usage_info.get("total_tokens")
    
    # Use actual balance difference if available, otherwise fall back to accumulated usage
    actual_credits_used = usage_info.get("actual_credits_used")
    accumulated_credits = usage_info.get("total_credits")
    credits_used = actual_credits_used if actual_credits_used is not None else accumulated_credits
    
    initial_balance = usage_info.get("initial_balance")
    final_balance = usage_info.get("final_balance")

    summary = {
        "run_id": run_id,
        "process_start": wall_start.isoformat(timespec="seconds"),
        "process_end": wall_end.isoformat(timespec="seconds"),
        "total_duration_seconds": round(process_end - process_start, 2),
        "annotation_phase_seconds": round(annotation_end - annotation_start, 2),
        "annotation_model": "openrouter consensus (mixed free models)",
        "tokens_used": total_tokens,
        "credits_used": credits_used,
        "credits_from_balance_check": actual_credits_used,
        "credits_from_api_responses": accumulated_credits,
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tissue": tissue_name,
    }

    adata.write(output_path)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\\nRun summary")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
"""
)


def find_all_h5ad_files(data_base_dir: str, h5ad_subdir: str) -> List[Tuple[str, str, str]]:
    """Return (tissue, file_path, dataset_name) tuples for all h5ad files."""
    all_files: List[Tuple[str, str, str]] = []
    data_path = Path(data_base_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_base_dir}")
        return []

    skip_dirs = {"metadata_processing", "__pycache__", ".git", ".ipynb_checkpoints"}

    for tissue_dir in sorted(data_path.iterdir()):
        if not tissue_dir.is_dir() or tissue_dir.name in skip_dirs:
            continue

        h5ad_dir = tissue_dir / h5ad_subdir
        if not h5ad_dir.exists():
            continue

        for h5ad_file in sorted(h5ad_dir.glob("*.h5ad")):
            dataset_name = h5ad_file.stem
            all_files.append((tissue_dir.name, str(h5ad_file), dataset_name))

    return all_files


def create_run_id(tissue_name: str, dataset_name: str, task: str = "celltype_annotation") -> str:
    clean_name = dataset_name.replace("Data_", "")
    tissue_suffix = f"_{tissue_name.capitalize()}"
    if clean_name.endswith(tissue_suffix):
        clean_name = clean_name[: -len(tissue_suffix)]
    clean_name = clean_name.replace("_", "").lower()
    return f"{tissue_name}_{clean_name}_{task}"


def build_temp_runner(temp_path: Path, h5ad_path: str, tissue: str, output_path: Path, summary_path: Path, run_id: str) -> Path:
    script_text = RUNNER_TEMPLATE.substitute(
        input_path=h5ad_path,
        tissue=tissue,
        output_path=str(output_path),
        summary_path=str(summary_path),
        run_id=run_id,
    )
    temp_path.write_text(script_text, encoding="utf-8")
    return temp_path


def save_batch_summary(results, output_file: Path) -> Path:
    summary = {
        "batch_start": results[0]["start_time"] if results else None,
        "batch_end": datetime.now().isoformat(),
        "total_runs": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "runs": results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mLLMCelltype annotator on all h5ad files across tissues")
    parser.add_argument("--data-dir", type=str, default=DATA_BASE_DIR, help="Base data directory containing tissue folders")
    parser.add_argument("--h5ad-subdir", type=str, default=DEFAULT_H5AD_SUBDIR, help="Subdirectory under each tissue containing h5ad files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to write annotated h5ad files and summaries")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without running models")
    parser.add_argument("--tissues", nargs="+", default=None, help="Specific tissues to process (default: all)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs if summary already exists")
    parser.add_argument("--start-from", type=str, default=None, help="Start from specific dataset (format: 'tissue/dataset' or just 'dataset')")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning for h5ad files in: {args.data_dir} (subdir: {args.h5ad_subdir})")
    all_files = find_all_h5ad_files(args.data_dir, args.h5ad_subdir)
    if not all_files:
        print("[WARN] No h5ad files found!")
        sys.exit(1)

    if args.tissues:
        all_files = [(t, f, d) for t, f, d in all_files if t in args.tissues]
        print(f"[INFO] Filtering to tissues: {args.tissues}")

    if not all_files:
        print("[WARN] No h5ad files remain after filtering")
        sys.exit(1)

    # Handle --start-from parameter
    start_tissue = None
    start_dataset = None
    if args.start_from:
        if "/" in args.start_from:
            start_tissue, start_dataset = args.start_from.split("/", 1)
        else:
            start_dataset = args.start_from
        
        # Find the starting point and filter files
        found_start = False
        filtered_files = []
        for tissue, filepath, dataset in all_files:
            # Check if this is our starting point
            if not found_start:
                dataset_match = (dataset == start_dataset or 
                               dataset.lower() == start_dataset.lower() or
                               start_dataset.lower() in dataset.lower())
                tissue_match = (start_tissue is None or tissue == start_tissue)
                
                if dataset_match and tissue_match:
                    found_start = True
                    print(f"[INFO] Starting from: {tissue}/{dataset}")
                    filtered_files.append((tissue, filepath, dataset))
                    continue
            else:
                filtered_files.append((tissue, filepath, dataset))
        
        if not found_start:
            print(f"[ERROR] Could not find starting point: {args.start_from}")
            print(f"[INFO] Available datasets:")
            for tissue, _, dataset in all_files:
                print(f"  {tissue}/{dataset}")
            sys.exit(1)
        
        all_files = filtered_files
        print(f"[INFO] Skipped to dataset '{args.start_from}' - {len(all_files)} file(s) remaining\n")

    print(f"[INFO] Found {len(all_files)} file(s) across {len(set(t for t, _, _ in all_files))} tissue(s)\n")

    by_tissue = {}
    for tissue, _, dataset in all_files:
        by_tissue.setdefault(tissue, []).append(dataset)
    for tissue, datasets in sorted(by_tissue.items()):
        print(f"  {tissue}: {len(datasets)} file(s)")
    print()

    if args.dry_run:
        print("[INFO] DRY RUN MODE - No model calls will be executed\n")

    results = []

    for idx, (tissue, h5ad_path, dataset_name) in enumerate(all_files, 1):
        print(f"\n{'=' * 70}")
        print(f"Processing {idx}/{len(all_files)}: {tissue}/{dataset_name}")
        print(f"{'=' * 70}")

        run_id = create_run_id(tissue, dataset_name)
        output_path = output_dir / f"{run_id}.h5ad"
        summary_path = output_dir / f"{run_id}_RUN_SUMMARY.json"
        temp_script_path = script_dir / f"_temp_run_{run_id}.py"

        if args.skip_existing and summary_path.exists():
            print(f"  [SKIP] Summary already exists: {summary_path}")
            results.append(
                {
                    "tissue": tissue,
                    "dataset": dataset_name,
                    "run_id": run_id,
                    "file_path": h5ad_path,
                    "success": True,
                    "status": "skipped",
                    "return_code": 0,
                    "start_time": None,
                }
            )
            continue

        start_time = datetime.now().isoformat()
        build_temp_runner(temp_script_path, h5ad_path, tissue, output_path, summary_path, run_id)

        if args.dry_run:
            print(f"  [DRY-RUN] Would run: python {temp_script_path}")
            status = "dry-run"
            success = True
            return_code = 0
        else:
            try:
                print(f"  [Running] python {temp_script_path}")
                result = subprocess.run(
                    ["python", str(temp_script_path)],
                    cwd=script_dir,
                    capture_output=False,
                    text=True,
                )
                success = result.returncode == 0
                return_code = result.returncode
                status = "completed" if success else "failed"
            except Exception as exc:  # pragma: no cover - subprocess failures
                success = False
                return_code = -1
                status = f"error: {exc}"
                print(f"  [ERROR] Failed to run {temp_script_path}: {exc}")

        if temp_script_path.exists():
            temp_script_path.unlink()

        results.append(
            {
                "tissue": tissue,
                "dataset": dataset_name,
                "run_id": run_id,
                "file_path": h5ad_path,
                "success": success,
                "return_code": return_code,
                "status": status,
                "start_time": start_time,
                "summary_path": str(summary_path),
                "output_path": str(output_path),
            }
        )

        if success:
            print("  ✓ Completed successfully")
        else:
            print(f"  ✗ Failed with code {return_code}")

    batch_summary_path = output_dir / "BATCH_SUMMARY.json"
    save_batch_summary(results, batch_summary_path)
    print(f"\n{'=' * 70}")
    print("BATCH SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total runs:    {len(results)}")
    print(f"Successful:    {sum(1 for r in results if r['success'])}")
    print(f"Failed:        {sum(1 for r in results if not r['success'])}")
    print(f"Summary saved: {batch_summary_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()


# python run_all_tissues.py --tissues brain --h5ad-subdir h5ad_unlabelled --output-dir output

# python run_all_tissues.py --tissues breast colorectal head_neck hematologic kidney --h5ad-subdir h5ad_unlabelled --output-dir output --start-from liu2021 --skip-existing


# python run_all_tissues.py --tissues liver lung neuroendcrine ovarian pancreas prostate scarcoma skin --h5ad-subdir h5ad_unlabelled_clustered --output-dir output 