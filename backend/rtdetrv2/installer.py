"""Installer helpers for the upstream lyuwenyu/RT-DETR repository."""
from __future__ import annotations

import subprocess
from pathlib import Path


def repo_dir() -> Path:
    from app.config import DATA_DIR

    return DATA_DIR / "vendor" / "RT-DETR"


def pytorch_root() -> Path:
    return repo_dir() / "rtdetrv2_pytorch"


def is_installed() -> bool:
    root = pytorch_root()
    return (root / "tools" / "train.py").exists() and (root / "src" / "core").exists()


def ensure_installed(log_fn=None) -> Path:
    """Clone the upstream repo if needed and return ``rtdetrv2_pytorch`` root."""
    repo = repo_dir()
    if is_installed():
        return pytorch_root()

    repo.parent.mkdir(parents=True, exist_ok=True)
    if repo.exists() and not (repo / ".git").exists():
        raise RuntimeError(f"RT-DETR vendor path exists but is not a git clone: {repo}")

    if repo.exists():
        cmd = ["git", "-C", str(repo), "pull", "--ff-only"]
    else:
        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/lyuwenyu/RT-DETR",
            str(repo),
        ]

    if log_fn:
        log_fn(f"RT-DETRv2 installer: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout and log_fn:
        for line in proc.stdout.splitlines():
            log_fn(f"RT-DETRv2 installer: {line}")
    if proc.returncode != 0:
        raise RuntimeError(f"RT-DETR clone/update failed with code {proc.returncode}")
    if not is_installed():
        raise RuntimeError(f"RT-DETRv2 PyTorch files not found after install: {pytorch_root()}")
    return pytorch_root()
