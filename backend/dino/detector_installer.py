"""Installer helpers for IDEA-Research/DINO detector."""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def repo_dir() -> Path:
    from app.config import DATA_DIR

    return DATA_DIR / "vendor" / "DINO-DETR"


def is_installed() -> bool:
    root = repo_dir()
    return (root / "main.py").exists() and (root / "config" / "DINO" / "DINO_4scale.py").exists()


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return
    path.write_text(text.replace(old, new), encoding="utf-8")


def _apply_compat_patches(root: Path) -> None:
    """Patch upstream for current PyTorch/YAPF without forking the repo."""
    _replace(
        root / "models/dino/ops/src/cuda/ms_deform_attn_cuda.cu",
        "AT_DISPATCH_FLOATING_TYPES(value.type(), \"ms_deform_attn_forward_cuda\"",
        "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), \"ms_deform_attn_forward_cuda\"",
    )
    _replace(
        root / "models/dino/ops/src/cuda/ms_deform_attn_cuda.cu",
        "AT_DISPATCH_FLOATING_TYPES(value.type(), \"ms_deform_attn_backward_cuda\"",
        "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), \"ms_deform_attn_backward_cuda\"",
    )
    _replace(
        root / "main.py",
        "torch.load(args.frozen_weights, map_location='cpu')",
        "torch.load(args.frozen_weights, map_location='cpu', weights_only=False)",
    )
    _replace(
        root / "main.py",
        "torch.load(args.resume, map_location='cpu')",
        "torch.load(args.resume, map_location='cpu', weights_only=False)",
    )
    _replace(
        root / "main.py",
        "torch.load(args.pretrain_model_path, map_location='cpu')['model']",
        "torch.load(args.pretrain_model_path, map_location='cpu', weights_only=False)['model']",
    )
    _replace(
        root / "util/slconfig.py",
        "text, _ = FormatCode(text, style_config=yapf_style, verify=True)",
        "try:\n            text, _ = FormatCode(text, style_config=yapf_style, verify=True)\n        except TypeError:\n            text, _ = FormatCode(text, style_config=yapf_style)",
    )
    _replace(
        root / "engine.py",
        "                # img_h, img_w = tgt['orig_size'].unbind()\n"
        "                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)\n"
        "                # _res_bbox = res['boxes'] / scale_fct\n"
        "                _res_bbox = outbbox",
        "                img_h, img_w = tgt['orig_size'].unbind()\n"
        "                scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)\n"
        "                res_xyxy = res['boxes'] / scale_fct\n"
        "                _res_bbox = torch.stack((\n"
        "                    (res_xyxy[:, 0] + res_xyxy[:, 2]) / 2,\n"
        "                    (res_xyxy[:, 1] + res_xyxy[:, 3]) / 2,\n"
        "                    res_xyxy[:, 2] - res_xyxy[:, 0],\n"
        "                    res_xyxy[:, 3] - res_xyxy[:, 1],\n"
        "                ), dim=1)",
    )


def _ops_import_ok(root: Path) -> bool:
    code = "import torch; import MultiScaleDeformableAttention; from models.dino.ops.modules import MSDeformAttn"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode == 0


def _build_ops(root: Path, log_fn=None) -> None:
    if _ops_import_ok(root):
        return
    cmd = [sys.executable, "setup.py", "build", "install"]
    if log_fn:
        log_fn(f"DINO detector ops build: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(root / "models/dino/ops"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout and log_fn:
        for line in proc.stdout.splitlines()[-80:]:
            log_fn(f"DINO detector ops: {line}")
    if proc.returncode != 0:
        raise RuntimeError(f"DINO detector CUDA op build failed with code {proc.returncode}: {(proc.stdout or '')[-2000:]}")
    if not _ops_import_ok(root):
        raise RuntimeError("DINO detector CUDA op was built but cannot be imported")


def ensure_installed(log_fn=None, build_ops: bool = True) -> Path:
    repo = repo_dir()
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not is_installed():
        if repo.exists() and not (repo / ".git").exists():
            raise RuntimeError(f"DINO detector vendor path exists but is not a git clone: {repo}")
        cmd = (
            ["git", "-C", str(repo), "pull", "--ff-only"]
            if repo.exists()
            else ["git", "clone", "--depth", "1", "https://github.com/IDEA-Research/DINO", str(repo)]
        )
        if log_fn:
            log_fn(f"DINO detector installer: {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.stdout and log_fn:
            for line in proc.stdout.splitlines():
                log_fn(f"DINO detector installer: {line}")
        if proc.returncode != 0:
            raise RuntimeError(f"DINO detector clone/update failed with code {proc.returncode}: {(proc.stdout or '')[-2000:]}")
    _apply_compat_patches(repo)
    if build_ops:
        _build_ops(repo, log_fn=log_fn)
    return repo
