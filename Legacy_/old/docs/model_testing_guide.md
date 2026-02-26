# Model Testing Guide

This guide explains how to evaluate the performance of your models in Model DESIGNER.

## 1. Classification Models

For classification tasks (e.g., MNIST, CIFAR-10), we evaluate how accurately the model predicts the correct category.

### 1.1 Metrics

- **Accuracy**: The percentage of correct predictions.
  - _Formula_: `(True Positives + True Negatives) / Total Samples`
  - _Goal_: High is better (close to 1.0 or 100%).
- **Loss (CrossEntropy)**: Measures the error between predicted probabilities and actual labels.
  - _Goal_: Low is better (converging to 0).

### 1.2 Evaluation Tools

1.  **Training Curves** (Dashboard):
    - **Loss Curve**: Should decrease rapidly and then stabilize. If it goes up, learning rate might be too high.
    - **Accuracy Curve**: Should increase and stabilize.
    - _Overfitting Check_: If Training Accuracy is high (99%) but Validation Accuracy is low (80%), the model is memorizing data instead of learning patterns. Use **Dropout** or **BatchNorm** to fix this.

2.  **Confusion Matrix** (Planned Feature):
    - A table showing predictions vs. actual labels.
    - _Diagonal_: Correct predictions.
    - _Off-diagonal_: Errors (e.g., confusing "3" with "8").

---

## 2. Object Detection Models

For detection tasks (e.g., COCO, YOLO-style), evaluation is more complex because we predict **What** (Class) and **Where** (Box).

### 2.1 Key Concepts

- **Bounding Box (BBox)**: The rectangle `[x, y, w, h]` enclosing an object.
- **IoU (Intersection over Union)**: Measures overlap between Predicted Box and Ground Truth Box.
  - _IoU = 1.0_: Perfect match.
  - _IoU > 0.5_: Usually considered a "True Positive".
  - _IoU < 0.5_: "False Positive" (bad localization).

### 2.2 Metrics

- **Precision**: Of all boxes predicted as "Car", how many are actually cars?
- **Recall**: Of all actual "Cars" in the image, how many did we find?
- **mAP (mean Average Precision)**: The gold standard metric.
  - It calculates Average Precision (Area under Precision-Recall curve) for each class and averages them.
  - **mAP@50**: mAP calculated at IoU threshold 0.5.
  - **mAP@50:95**: Average mAP over IoU thresholds 0.50 to 0.95 (stricter).

### 2.3 Visual Inspection (NMS)

Raw output from a detection model produces thousands of overlapping boxes. We use **NMS (Non-Maximum Suppression)** to clean this up:

1.  Sort boxes by confidence score.
2.  Pick the highest confidence box.
3.  Discard all other boxes that have high IoU (overlap) with it.
4.  Repeat.

_Visual Test_: Check if the model detects one clear box per object, rather than multiple jittery boxes.
