# Model Creation Guide & Gap Analysis

## 1. Classification (Currently Supported)

### Step-by-Step Guide

1.  **Project Setup**:
    - Open Model DESIGNER.
    - Drag an **Input** node.
    - Set parameters: `Channels=1`, `Height=28`, `Width=28` (for MNIST).
    - _Note: For CIFAR-10, use Channels=3, 32x32._

2.  **Feature Extraction (Backbone)**:
    - Add **Conv2d** nodes (e.g., 32 filters, 3x3 kernel).
    - Add **ReLU** activations after Conv2d.
    - Add **MaxPool2d** to reduce spatial dimensions (e.g., 2x2).
    - Repeat this block 2-3 times.

3.  **Classification Head**:
    - Add a **Flatten** node to convert 2D feature maps to 1D vector.
    - Add **Linear** (Dense) nodes.
    - _Optional: Add **Dropout** for regularization._
    - Add an **Output** node validation.
    - Set `Num Classes=10` (for MNIST/CIFAR).

4.  **Build & Train**:
    - Click **Build** to generate PyTorch code.
    - Go to **Dashboard** or **Jobs**.
    - Start Training:
      - Select Dataset: `MNIST`.
      - Set Epochs: ~5-10.
      - Batch Size: 32 or 64.
    - Monitor Accuracy and Loss in real-time.

---

## 2. Object Detection (Experimental Support)

**Status**: 🛠️ **Design & Code Gen Supported** (Training backend pending Phase 2)

You can now use **Upsample** and **Concatenate** layers to build detection architectures (like U-Net or FPN).

### Step-by-Step Design Guide

1.  **Backbone (Feature Extractor)**:
    - Add **Input** node (e.g., 3x640x640 for COCO).
    - Add a sequence of **Conv2d** -> **ReLU** -> **MaxPool2d** blocks.
    - _Example_: Create 3 blocks. The output of Block 1 is "High Res", Block 3 is "Low Res".

2.  **Neck (Feature Fusion)**:
    - Add an **Upsample** node after Block 3. Set `Scale Factor=2`.
    - Add a **Concatenate** node.
    - Connect the **Upsample** output to the Concatenate node.
    - Connect the **Block 2** output to the Concatenate node (Skip Connection).
    - Set `Concatenate` param `Dim=1` (Channel dimension).

3.  **Head (Prediction)**:
    - Add a **Conv2d** node (Kernel=1) after the Concatenate node.
    - Set `Out Channels` to `(Classes + 5) * Anchors`.
    - Connect to **Output** node.

---

## 3. Implementation Status & Gaps

| Feature       | Status         | Notes                                                                                            |
| :------------ | :------------- | :----------------------------------------------------------------------------------------------- |
| **Layers**    | ✅ **Ready**   | `Upsample` and `Concatenate` added in Feature E.                                                 |
| **Code Gen**  | ✅ **Ready**   | Backend generates valid PyTorch code for branching graphs.                                       |
| **Data**      | ⚠️ **Partial** | COCO support added but requires `pycocotools` and downloaded data.                               |
| **Training**  | ❌ **Missing** | Loss function is hardcoded to CrossEntropy (Classification). Needs **Detection Loss** (Phase 2). |
| **Inference** | ❌ **Missing** | No Bounding Box visualization yet.                                                               |

---

## Further Reading

- [Testing Guide](./model_testing_guide.md) - Learn how to evaluate your metrics.
