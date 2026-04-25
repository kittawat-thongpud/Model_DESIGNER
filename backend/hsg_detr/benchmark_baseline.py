"""
Baseline comparison: HSG-DETR-n vs YOLOv8-RTDETR-n
Trains both from scratch on COCO128 for 3 epochs each, compares mAP.
"""
import sys
sys.path.insert(0, '/home/rase/kittawat_ws/Model_DESIGNER/backend')

import time
import json
from pathlib import Path

import hsg_detr  # noqa: F401 — registers custom modules
from ultralytics import RTDETR


def train_and_validate(model_yaml: str, name: str, epochs: int = 3):
    """Train one model, return final metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    model = RTDETR(model_yaml)

    t0 = time.time()
    results = model.train(
        data='coco128.yaml',
        epochs=epochs,
        imgsz=640,
        batch=4,
        verbose=False,
        patience=0,            # no early stopping for short run
        exist_ok=True,
        name=name,
        project='runs/detect/bench',
    )
    train_time = time.time() - t0

    # Best validation from training
    maps = results.results_dict
    mAP50 = maps.get('metrics/mAP50(B)', 0.0)
    mAP50_95 = maps.get('metrics/mAP50-95(B)', 0.0)

    print(f"  Epochs: {epochs}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"  mAP50:      {mAP50:.4f}")
    print(f"  mAP50-95:   {mAP50_95:.4f}")

    return {
        'name': name,
        'epochs': epochs,
        'train_time_s': train_time,
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
    }


def main():
    epochs = 3
    print(f"Running baseline comparison: {epochs} epochs on COCO128")

    results = []
    results.append(train_and_validate(
        '/home/rase/kittawat_ws/Model_DESIGNER/backend/hsg_detr/configs/hsg_detr_n.yaml',
        'hsg_detr_n',
        epochs,
    ))
    results.append(train_and_validate(
        'yolov8-rtdetr.yaml',
        'yolov8_rtdetr_n',
        epochs,
    ))

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['name']:20s} | mAP50: {r['mAP50']:.4f} | mAP50-95: {r['mAP50_95']:.4f} | time: {r['train_time_s']:.0f}s")

    diff_map50 = results[0]['mAP50'] - results[1]['mAP50']
    diff_map5095 = results[0]['mAP50_95'] - results[1]['mAP50_95']
    print(f"\n  HSG-DETR vs baseline delta:")
    print(f"    mAP50     : {diff_map50:+.4f} ({diff_map50 / max(results[1]['mAP50'], 1e-6) * 100:+.1f}%)")
    print(f"    mAP50-95  : {diff_map5095:+.4f} ({diff_map5095 / max(results[1]['mAP50_95'], 1e-6) * 100:+.1f}%)")

    out = Path('runs/detect/bench/comparison.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == '__main__':
    main()
