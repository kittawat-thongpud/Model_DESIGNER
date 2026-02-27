"""
Package Service — export/import .mdpkg bundles.

A .mdpkg file is a ZIP archive containing:
  manifest.json          — package metadata & inventory
  weights/<weight_id>/
    meta.json            — weight metadata
    weight.pt            — trained weights file
  jobs/<job_id>/
    record.json          — job record
    data.yaml            — dataset config used during training
    extended_metrics.jsonl  (optional)
    log.jsonl               (optional)

Export collects the full lineage chain:
  - For a weight: follows parent_weight_id chain (oldest → newest),
    plus all linked job records.
  - For a job: includes the job's output weight, then follows that
    weight's lineage (which may include pretrained parent weights).
"""
from __future__ import annotations

import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone

from ..config import JOBS_DIR as _JOB_DIR, WEIGHTS_DIR as _WEIGHTS_DIR
from .job_storage import _job_dir, load_job, _store as _job_store
from .weight_storage import load_weight_meta, weight_pt_path, list_weights, _store as _weight_store

PACKAGE_VERSION = "1"
PACKAGE_MIME = "application/x-mdpkg"
PACKAGE_EXT = ".mdpkg"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_weight_lineage(weight_id: str) -> list[str]:
    """Walk parent_weight_id chain, return list oldest→newest."""
    chain: list[str] = []
    visited: set[str] = set()
    current = weight_id
    while current and current not in visited:
        visited.add(current)
        chain.append(current)
        meta = load_weight_meta(current)
        if not meta:
            break
        current = meta.get("parent_weight_id") or ""
    chain.reverse()   # oldest first
    return chain


def _collect_job_ids_for_weights(weight_ids: list[str]) -> list[str]:
    """Return unique job_ids referenced by the given weights (preserving order)."""
    seen: set[str] = set()
    result: list[str] = []
    for wid in weight_ids:
        meta = load_weight_meta(wid)
        if not meta:
            continue
        # job_id from meta + job_id from each training_run
        jids = []
        if meta.get("job_id"):
            jids.append(meta["job_id"])
        for run in meta.get("training_runs", []):
            if run.get("job_id"):
                jids.append(run["job_id"])
        for jid in jids:
            if jid and jid not in seen:
                seen.add(jid)
                result.append(jid)
    return result


# ─── BUILD package ────────────────────────────────────────────────────────────

def build_weight_package(weight_id: str, include_jobs: bool = False) -> tuple[bytes, str]:
    """
    Build a .mdpkg for a weight (including full lineage).
    include_jobs=False omits training job records to reduce file size.
    Returns (zip_bytes, suggested_filename).
    """
    weight_ids = _collect_weight_lineage(weight_id)
    job_ids = _collect_job_ids_for_weights(weight_ids) if include_jobs else []

    root_meta = load_weight_meta(weight_id)
    model_name = (root_meta or {}).get("model_name", weight_id[:8])
    filename = f"{model_name}_{weight_id[:8]}{PACKAGE_EXT}".replace(" ", "_")

    manifest = {
        "version": PACKAGE_VERSION,
        "type": "weight",
        "root_weight_id": weight_id,
        "created_at": _now_iso(),
        "weights": weight_ids,
        "jobs": job_ids,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        _pack_weights(zf, weight_ids)
        if include_jobs:
            _pack_jobs(zf, job_ids)
    return buf.getvalue(), filename


def build_job_package(job_id: str, include_jobs: bool = False) -> tuple[bytes, str]:
    """
    Build a .mdpkg for a job (includes output weight lineage).
    include_jobs=False omits training job records to reduce file size.
    Returns (zip_bytes, suggested_filename).
    """
    record = load_job(job_id)
    if not record:
        raise FileNotFoundError(f"Job '{job_id}' not found")

    output_weight_id: str | None = record.get("weight_id")
    if output_weight_id:
        weight_ids = _collect_weight_lineage(output_weight_id)
    else:
        weight_ids = []

    if include_jobs:
        job_ids_from_weights = _collect_job_ids_for_weights(weight_ids)
        all_job_ids = list(dict.fromkeys([job_id] + job_ids_from_weights))
    else:
        all_job_ids = []

    model_name = record.get("model_name", job_id[:8])
    filename = f"job_{model_name}_{job_id[:8]}{PACKAGE_EXT}".replace(" ", "_")

    manifest = {
        "version": PACKAGE_VERSION,
        "type": "job",
        "root_job_id": job_id,
        "created_at": _now_iso(),
        "weights": weight_ids,
        "jobs": all_job_ids,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        _pack_weights(zf, weight_ids)
        if include_jobs:
            _pack_jobs(zf, all_job_ids)
    return buf.getvalue(), filename


def _pack_weights(zf: zipfile.ZipFile, weight_ids: list[str]) -> None:
    for wid in weight_ids:
        meta_path = _WEIGHTS_DIR / wid / "meta.json"
        pt_path = weight_pt_path(wid)
        if meta_path.exists():
            zf.write(meta_path, f"weights/{wid}/meta.json")
        if pt_path.exists():
            zf.write(pt_path, f"weights/{wid}/weight.pt")


def _pack_jobs(zf: zipfile.ZipFile, job_ids: list[str]) -> None:
    for jid in job_ids:
        job_dir = _job_dir(jid)
        if not job_dir.exists():
            continue
        for fname in ("record.json", "data.yaml", "extended_metrics.jsonl", "log.jsonl"):
            fpath = job_dir / fname
            if fpath.exists():
                zf.write(fpath, f"jobs/{jid}/{fname}")


# ─── IMPORT package ───────────────────────────────────────────────────────────

def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class ImportResult:
    def __init__(self) -> None:
        self.weights_imported: list[dict] = []   # [{old_id, new_id, name}]
        self.jobs_imported: list[dict] = []       # [{old_id, new_id}]
        self.errors: list[str] = []

    def to_dict(self) -> dict:
        return {
            "weights_imported": self.weights_imported,
            "jobs_imported": self.jobs_imported,
            "errors": self.errors,
        }


def peek_package(data: bytes) -> dict:
    """
    Read manifest from a .mdpkg without importing.
    Returns {weights: [{id, name, model_name, dataset}], jobs: [id, ...]}.
    """
    try:
        buf = io.BytesIO(data)
        with zipfile.ZipFile(buf, "r") as zf:
            if "manifest.json" not in zf.namelist():
                return {"error": "manifest.json not found"}
            manifest = json.loads(zf.read("manifest.json"))
            if manifest.get("version") != PACKAGE_VERSION:
                return {"error": f"Unsupported version: {manifest.get('version')}"}

            weights_info = []
            for wid in manifest.get("weights", []):
                meta_arc = f"weights/{wid}/meta.json"
                meta: dict = {}
                if meta_arc in zf.namelist():
                    meta = json.loads(zf.read(meta_arc))
                # Resolve a clean dataset label: prefer dataset_name, then derive from dataset path.
                # If path ends with a .yaml/.yml file (e.g. /data/coco128/data.yaml), use the parent
                # folder name (coco128) as the label rather than the filename.
                raw_dataset = meta.get("dataset_name") or meta.get("dataset", "")
                if raw_dataset:
                    bn = os.path.basename(raw_dataset.rstrip("/\\"))
                    if bn.lower().endswith((".yaml", ".yml")):
                        dataset_label = os.path.basename(os.path.dirname(raw_dataset.rstrip("/\\")))
                    else:
                        dataset_label = bn
                else:
                    dataset_label = ""
                weights_info.append({
                    "id": wid,
                    "model_name": meta.get("model_name", wid[:8]),
                    "dataset": dataset_label,
                    "epochs_trained": meta.get("epochs_trained", 0),
                })
            return {
                "version": manifest.get("version"),
                "weights": weights_info,
                "jobs": manifest.get("jobs", []),
            }
    except zipfile.BadZipFile:
        return {"error": "Not a valid ZIP / .mdpkg archive"}


def import_package(
    data: bytes,
    rename_map: dict[str, str] | None = None,
    include_jobs: bool = False,
) -> ImportResult:
    """
    Import a .mdpkg file. Always assigns NEW IDs — never clashes with existing data.
    rename_map: {old_weight_id: new_display_name} — overrides model_name in meta.
    include_jobs: if False (default), skip importing job records.
    """
    result = ImportResult()
    rename_map = rename_map or {}

    try:
        buf = io.BytesIO(data)
        with zipfile.ZipFile(buf, "r") as zf:
            names = set(zf.namelist())

            if "manifest.json" not in names:
                result.errors.append("manifest.json not found — not a valid .mdpkg")
                return result

            manifest = json.loads(zf.read("manifest.json"))
            if manifest.get("version") != PACKAGE_VERSION:
                result.errors.append(f"Unsupported package version: {manifest.get('version')}")
                return result

            # Build old→new ID maps upfront so we can rewrite cross-references
            weight_id_map: dict[str, str] = {wid: _new_id() for wid in manifest.get("weights", [])}
            job_id_map: dict[str, str] = {jid: _new_id() for jid in manifest.get("jobs", [])}

            # Import weights (ordered: lineage oldest→newest in manifest)
            for old_wid in manifest.get("weights", []):
                try:
                    _import_weight(
                        zf, old_wid, weight_id_map, job_id_map,
                        rename_map.get(old_wid), result,
                    )
                except Exception as e:
                    result.errors.append(f"weight {old_wid}: {e}")

            # Import jobs (only if include_jobs=True)
            if include_jobs:
                for old_jid in manifest.get("jobs", []):
                    try:
                        _import_job(zf, old_jid, job_id_map, weight_id_map, result)
                    except Exception as e:
                        result.errors.append(f"job {old_jid}: {e}")

    except zipfile.BadZipFile:
        result.errors.append("File is not a valid ZIP / .mdpkg archive")

    return result


def _import_weight(
    zf: zipfile.ZipFile,
    old_wid: str,
    weight_id_map: dict[str, str],
    job_id_map: dict[str, str],
    new_name: str | None,
    result: ImportResult,
) -> None:
    meta_arc = f"weights/{old_wid}/meta.json"
    pt_arc = f"weights/{old_wid}/weight.pt"
    names = set(zf.namelist())

    if meta_arc not in names:
        result.errors.append(f"weight {old_wid}: meta.json missing in package")
        return

    meta: dict = json.loads(zf.read(meta_arc))
    new_wid = weight_id_map[old_wid]

    # Rewrite weight_id
    meta["weight_id"] = new_wid

    # Rename display name if provided
    if new_name:
        meta["model_name"] = new_name

    # Rewrite parent_weight_id reference
    old_parent = meta.get("parent_weight_id")
    if old_parent and old_parent in weight_id_map:
        meta["parent_weight_id"] = weight_id_map[old_parent]

    # Rewrite job_id reference
    old_job = meta.get("job_id")
    if old_job and old_job in job_id_map:
        meta["job_id"] = job_id_map[old_job]

    # Rewrite weight_id / job_id inside training_runs
    for run in meta.get("training_runs", []):
        if run.get("weight_id") in weight_id_map:
            run["weight_id"] = weight_id_map[run["weight_id"]]
        if run.get("job_id") in job_id_map:
            run["job_id"] = job_id_map[run["job_id"]]

    # Write files — use _weight_store.save() so _index.json is updated
    dest_dir = _WEIGHTS_DIR / new_wid
    dest_dir.mkdir(parents=True, exist_ok=True)
    _weight_store.save(new_wid, meta)
    if pt_arc in names:
        (dest_dir / "weight.pt").write_bytes(zf.read(pt_arc))

    result.weights_imported.append({
        "old_id": old_wid,
        "new_id": new_wid,
        "name": meta.get("model_name", new_wid[:8]),
    })


def _import_job(
    zf: zipfile.ZipFile,
    old_jid: str,
    job_id_map: dict[str, str],
    weight_id_map: dict[str, str],
    result: ImportResult,
) -> None:
    record_arc = f"jobs/{old_jid}/record.json"
    names = set(zf.namelist())

    if record_arc not in names:
        result.errors.append(f"job {old_jid}: record.json missing in package")
        return

    record: dict = json.loads(zf.read(record_arc))
    new_jid = job_id_map[old_jid]

    # Rewrite job_id
    record["job_id"] = new_jid

    # Rewrite weight_id references inside job record
    for key in ("weight_id", "parent_weight_id", "output_weight_id"):
        old_val = record.get(key)
        if old_val and old_val in weight_id_map:
            record[key] = weight_id_map[old_val]

    dest_dir = _job_dir(new_jid)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _job_store.save(new_jid, record)

    for fname in ("data.yaml", "extended_metrics.jsonl", "log.jsonl"):
        arc = f"jobs/{old_jid}/{fname}"
        if arc in names:
            (dest_dir / fname).write_bytes(zf.read(arc))

    result.jobs_imported.append({"old_id": old_jid, "new_id": new_jid})
