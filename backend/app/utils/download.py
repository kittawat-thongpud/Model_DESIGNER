"""
Shared download utilities — used by dataset plugins and the controller.

Provides progress-tracked file downloading and zip extraction.
"""
from __future__ import annotations
import os
import time
import urllib.request
import zipfile


def download_file(url: str, dest: str, state: dict, step_label: str,
                  file_key: str | None = None):
    """Download a single file with per-file progress tracking.

    Args:
        url: URL to download from.
        dest: Local file path to save to.
        state: Shared progress dict (written to by reporthook).
        step_label: Label prefix for messages (e.g. "[1/3]").
        file_key: Unique key for per-file progress in state["files"].
    """
    fname = os.path.basename(url)
    fk = file_key or fname

    if os.path.isfile(dest):
        state["message"] = f"{step_label}: {fname} already exists, skipping"
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Register per-file entry
    fentry = {
        "name": fname, "label": step_label, "status": "downloading",
        "progress": 0, "bytes_downloaded": 0, "bytes_total": 0,
        "rate_bps": 0, "eta_seconds": -1, "message": f"Downloading {fname}...",
    }
    state["files"][fk] = fentry
    state["current_file"] = fname
    state["message"] = f"{step_label}: Downloading {fname}..."
    file_start = time.monotonic()

    def _reporthook(count, block_size, total_size):
        downloaded = count * block_size
        elapsed = max(time.monotonic() - file_start, 0.01)
        rate = downloaded / elapsed

        fentry["rate_bps"] = rate
        fentry["bytes_downloaded"] = downloaded
        state["rate_bps"] = rate

        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            remaining = total_size - downloaded
            eta = remaining / rate if rate > 0 else -1
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            rate_mb = rate / (1024 * 1024)

            fentry["progress"] = round(pct, 1)
            fentry["bytes_total"] = total_size
            fentry["eta_seconds"] = round(eta, 1) if eta >= 0 else -1
            fentry["message"] = f"{mb_done:.1f}/{mb_total:.1f} MB ({rate_mb:.1f} MB/s)"

            state["progress"] = round(pct, 1)
            state["bytes_downloaded"] = downloaded
            state["bytes_total"] = total_size
            state["eta_seconds"] = round(eta, 1) if eta >= 0 else -1
            state["message"] = f"{step_label}: {fname} {mb_done:.1f}/{mb_total:.1f} MB ({rate_mb:.1f} MB/s)"
        else:
            mb_done = downloaded / (1024 * 1024)
            rate_mb = rate / (1024 * 1024)
            fentry["message"] = f"{mb_done:.1f} MB ({rate_mb:.1f} MB/s)"
            fentry["eta_seconds"] = -1
            state["message"] = f"{step_label}: {fname} {mb_done:.1f} MB ({rate_mb:.1f} MB/s)"
            state["eta_seconds"] = -1

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    fentry["progress"] = 100
    fentry["status"] = "done"
    fentry["message"] = "Done"


def tracked_torchvision_download(ds_cls, ds_root: str, state: dict):
    """Download a torchvision dataset with progress tracking.

    Monkey-patches torchvision's download_url to report progress into `state`.
    """
    import torchvision.datasets.utils as tv_utils

    orig_download = tv_utils.download_url

    def _tracked(url, root, filename=None, md5=None):
        fname = filename or os.path.basename(url.split("?")[0])
        fpath = os.path.join(root, fname)
        os.makedirs(root, exist_ok=True)
        if os.path.isfile(fpath):
            if md5 is None or tv_utils.check_integrity(fpath, md5):
                state["message"] = f"{fname} already exists, skipping"
                return
        state["current_file"] = fname
        state["message"] = f"Downloading {fname}..."
        dl_start = time.monotonic()

        def _hook(count, block_size, total_size):
            downloaded = count * block_size
            elapsed = max(time.monotonic() - dl_start, 0.01)
            rate = downloaded / elapsed
            state["rate_bps"] = rate
            if total_size > 0:
                pct = min(100.0, downloaded * 100.0 / total_size)
                remaining = total_size - downloaded
                eta = remaining / rate if rate > 0 else -1
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                rate_mb = rate / (1024 * 1024)
                state["progress"] = round(pct, 1)
                state["eta_seconds"] = round(eta, 1) if eta >= 0 else -1
                state["message"] = f"Downloading {fname}... {mb_done:.1f}/{mb_total:.1f} MB ({rate_mb:.1f} MB/s)"
            else:
                mb_done = downloaded / (1024 * 1024)
                rate_mb = rate / (1024 * 1024)
                state["message"] = f"Downloading {fname}... {mb_done:.1f} MB ({rate_mb:.1f} MB/s)"
                state["eta_seconds"] = -1

        urllib.request.urlretrieve(url, fpath, reporthook=_hook)
        if md5 is not None and not tv_utils.check_integrity(fpath, md5):
            raise RuntimeError(f"Integrity check failed for {fname}")

    tv_utils.download_url = _tracked
    try:
        ds_cls(ds_root, train=True, download=True)
    finally:
        tv_utils.download_url = orig_download


def download_and_extract(url: str, extract_to: str, state: dict,
                         step_label: str, file_key: str | None = None):
    """Download a zip, extract it, then delete the zip."""
    fname = os.path.basename(url)
    fk = file_key or fname
    zip_path = os.path.join(extract_to, fname)
    download_file(url, zip_path, state, step_label, file_key=fk)

    # Extract phase
    fentry = state["files"].get(fk)
    if fentry:
        fentry["status"] = "extracting"
        fentry["message"] = "Extracting..."
        fentry["progress"] = 0
    state["message"] = f"{step_label}: Extracting {fname}..."
    state["progress"] = 0
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        os.remove(zip_path)

    # Mark file complete and remove from active files
    if fk in state["files"]:
        del state["files"][fk]
