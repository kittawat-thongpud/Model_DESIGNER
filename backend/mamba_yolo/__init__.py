"""
Mamba-YOLO — SSM-based YOLO variant.

Package layout
--------------
mamba_yolo/
  installer.py   — clones HZAI-ZJNU/Mamba-YOLO repo + installs deps
  nn/            — loads real modules from cloned repo; patches parse_model
    __init__.py  — register() loads from vendor repo and patches ultralytics
  configs/       — Ultralytics-format YAML model definitions
"""
