#!/usr/bin/env python3
"""
analyze_top_performers.py
=========================
Analyze jobs from multiple servers and group top performers by (scale, imgsz).

Usage:
    python scripts/analyze_top_performers.py [jobs1.json jobs2.json ...] --output top_performers.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jobs_from_file(path: Path) -> list[dict]:
    """Load jobs from a JSON file (either direct list or {count, items} format)."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    return []


def extract_key_fields(job: dict) -> dict:
    """Extract relevant fields for comparison."""
    config = job.get("config", {})
    return {
        "job_id": job.get("job_id"),
        "model_name": job.get("model_name", "Unknown"),
        "model_id": job.get("model_id", ""),
        "model_scale": (job.get("model_scale") or "unknown").lower(),
        "imgsz": config.get("imgsz", 640),
        "batch": config.get("batch", 16),
        "epochs": job.get("epoch", 0),
        "total_epochs": job.get("total_epochs", 0),
        "status": job.get("status", "unknown"),
        "best_mAP50": job.get("best_mAP50") or 0,
        "best_mAP50_95": job.get("best_mAP50_95") or 0,
        "weight_id": job.get("weight_id"),
        "dataset": job.get("dataset_name", config.get("dataset_name", "unknown")),
        "optimizer": config.get("optimizer", "auto"),
        "lr0": config.get("lr0", 0.01),
        "server": job.get("_server", "unknown"),
    }


def group_by_scale_imgsz(jobs: list[dict]) -> dict[tuple, list[dict]]:
    """Group jobs by (scale, imgsz) tuple."""
    groups = defaultdict(list)
    for job in jobs:
        scale = job.get("model_scale", "unknown")
        imgsz = job.get("imgsz", 640)
        key = (scale, imgsz)
        groups[key].append(job)
    return dict(groups)


def find_top_performer(jobs: list[dict]) -> dict | None:
    """Find the best job by mAP50-95 among completed jobs."""
    completed = [j for j in jobs if j.get("status") == "completed" and j.get("best_mAP50_95", 0) > 0]
    if not completed:
        return None
    return max(completed, key=lambda x: x.get("best_mAP50_95", 0))


def main():
    parser = argparse.ArgumentParser(description="Analyze top performers by scale and imgsz")
    parser.add_argument("files", nargs="*", help="Job JSON files from dump_training_state.py")
    parser.add_argument("--output", default="./top_performers_analysis.json", help="Output JSON file")
    parser.add_argument("--min-epochs", type=int, default=50, help="Minimum epochs to consider valid")
    args = parser.parse_args()

    # Load all jobs
    all_jobs: list[dict] = []
    
    # If no files specified, try to load from training_state directory
    if not args.files:
        default_path = Path("./training_state/jobs.json")
        if default_path.exists():
            args.files = [str(default_path)]
    
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        jobs = load_jobs_from_file(path)
        # Tag with server name from path if available
        server_name = path.parent.name if "training_state" in str(path) else path.stem
        for j in jobs:
            j["_server"] = server_name
        all_jobs.extend(jobs)
        print(f"Loaded {len(jobs)} jobs from {path}")

    if not all_jobs:
        print("No jobs loaded!")
        return

    # Process jobs
    processed = [extract_key_fields(j) for j in all_jobs]
    
    # Group by (scale, imgsz)
    groups = group_by_scale_imgsz(processed)
    
    # Build analysis results
    results = {
        "summary": {
            "total_jobs": len(all_jobs),
            "completed_jobs": len([j for j in processed if j["status"] == "completed"]),
            "groups": len(groups),
            "scales": sorted(set(k[0] for k in groups.keys())),
            "imgszs": sorted(set(k[1] for k in groups.keys())),
        },
        "top_performers_by_group": {},
        "all_groups": {},
    }

    # Analyze each group
    for (scale, imgsz), jobs in sorted(groups.items()):
        key = f"{scale}_{imgsz}"
        
        # Filter valid jobs (completed with min epochs)
        valid = [j for j in jobs if j["status"] == "completed" and j["epochs"] >= args.min_epochs]
        
        # Sort by mAP50-95
        valid_sorted = sorted(valid, key=lambda x: x["best_mAP50_95"], reverse=True)
        
        # Get top performer
        top = find_top_performer(jobs)
        
        group_info = {
            "scale": scale,
            "imgsz": imgsz,
            "total_jobs": len(jobs),
            "valid_completed": len(valid),
            "all_jobs": jobs,  # All jobs in this group
            "ranked_jobs": valid_sorted,  # Sorted by mAP
        }
        
        if top:
            group_info["top_performer"] = top
            results["top_performers_by_group"][key] = {
                "scale": scale,
                "imgsz": imgsz,
                "job_id": top["job_id"],
                "model_name": top["model_name"],
                "model_id": top["model_id"],
                "best_mAP50": round(top["best_mAP50"], 5),
                "best_mAP50_95": round(top["best_mAP50_95"], 5),
                "epochs": f"{top['epochs']}/{top['total_epochs']}",
                "batch": top["batch"],
                "optimizer": top["optimizer"],
                "lr0": top["lr0"],
                "weight_id": top["weight_id"],
            }
        
        results["all_groups"][key] = group_info

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults written to: {out_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("TOP PERFORMERS BY (SCALE, IMGSZ)")
    print("="*80)
    print(f"{'Group':<15} {'Model':<30} {'mAP50':>8} {'mAP50-95':>10} {'Epochs':>8} {'Job ID':<16}")
    print("-"*80)
    
    for key, info in sorted(results["top_performers_by_group"].items()):
        model_name = info["model_name"][:28]
        print(f"{key:<15} {model_name:<30} {info['best_mAP50']:>8.4f} {info['best_mAP50_95']:>10.5f} {info['epochs']:>8} {info['job_id']:<16}")
    
    print("="*80)


if __name__ == "__main__":
    main()
