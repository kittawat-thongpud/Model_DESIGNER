"""Installer helpers for the upstream IDEA-Research/DINO-DETR repository."""
from __future__ import annotations

import subprocess
from pathlib import Path


def repo_dir() -> Path:
    from app.config import DATA_DIR

    return DATA_DIR / "vendor" / "DINO-DETR"


def is_installed() -> bool:
    root = repo_dir()
    return (root / "main.py").exists() and (root / "engine.py").exists()


def ensure_installed(log_fn=None) -> Path:
    """Clone/update upstream DINO-DETR if needed and return the repo root."""
    repo = repo_dir()
    if is_installed():
        return repo

    repo.parent.mkdir(parents=True, exist_ok=True)
    if repo.exists() and not (repo / ".git").exists():
        raise RuntimeError(f"DINO-DETR vendor path exists but is not a git clone: {repo}")

    if repo.exists():
        cmd = ["git", "-C", str(repo), "pull", "--ff-only"]
    else:
        cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/IDEA-Research/DINO-DETR",
            str(repo),
        ]

    if log_fn:
        log_fn(f"DINO-DETR installer: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout and log_fn:
        for line in proc.stdout.splitlines():
            log_fn(f"DINO-DETR installer: {line}")
    if proc.returncode != 0:
        raise RuntimeError(f"DINO-DETR clone/update failed with code {proc.returncode}")
    if not is_installed():
        raise RuntimeError(f"DINO-DETR files not found after install: {repo}")
    return repo
