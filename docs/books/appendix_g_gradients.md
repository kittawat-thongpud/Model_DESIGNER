# ภาคผนวก G --- Gradient Derivations

> _การหา gradient ของ loss function เป็นหัวใจสำหรับทำความเข้าใจว่าเครือข่ายเรียนรู้อะไร_

## G.1 BCE Gradient

### สูตร BCE (Binary Cross-Entropy)

$$
 L_{BCE} = -y\log p + (1-y)\log(1-p)
$$

### Gradient w.r.t. raw logit $z$ (before sigmoid)

เนื่องจาก $p = \sigma(z)$:

$$
 \frac{\partial L}{\partial z} = p - y
$$

**ข้อสังเกต:**

- ถ้า $y = 1, p = 0.9 \Rightarrow \frac{\partial L}{\partial z} = -0.1$ (push logit up)
- ถ้า $y = 0, p = 0.1 \Rightarrow \frac{\partial L}{\partial z} = 0.1$ (push logit down)
- **Gradient magnitude เป็นสัดส่วนกับ error**

## G.2 Focal Loss Gradient

### สูตร Focal Loss

$$
 L_{FL} = -\alpha_t (1-p_t)^\gamma \log(p_t)
$$

### Gradient

$$
 \frac{\partial L_{FL}}{\partial z} = \alpha_t (1-p_t)^\gamma (p_t - y) + \alpha_t \gamma (1-p_t)^{\gamma-1} p_t \log(p_t)(p_t - y)
$$

**นัยยะเชิงสถาปัตยกรรม:**

- Easy examples (high $p_t$) → $(1-p_t)^\gamma \approx 0$ → gradient ถูก suppress
- Hard examples (low $p_t$) → gradient สูง → เครือข่ายโฟกัสที่ตัวอย่างยาก
- **$\gamma = 2$ คือค่าที่พบว่าดีที่สุดใน practice**

## G.3 CIoU Gradient

### สูตร CIoU

$$
 L_{CIoU} = 1 - \text{IoU} + \frac{\rho^2}{c^2} + \alpha v
$$

### Gradient w.r.t. predicted box parameters

$$
 \frac{\partial L}{\partial b_x} = -\frac{\partial \text{IoU}}{\partial b_x} + \frac{2(b_x - b_x^{gt})}{c^2}
$$

$$
 \frac{\partial L}{\partial b_w} = -\frac{\partial \text{IoU}}{\partial b_w} + \alpha \frac{\partial v}{\partial b_w}
$$

โดยที่ aspect ratio gradient:

$$
 \frac{\partial v}{\partial b_w} = \frac{8}{\pi^2}\left(\arctan\frac{w_{gt}}{h_{gt}} - \arctan\frac{w}{h}\right) \cdot \frac{h}{w^2+h^2}
$$

**ข้อสำคัญ:**

- CIoU ให้ gradient ที่มี 3 components: **overlap**, **center distance**, **aspect ratio**
- เมื่อ $\text{IoU} = 0 \rightarrow$ gradient ยังคงมีจาก distance term (ไม่เหมือน IoU loss ปกติ)

## G.4 DFL Gradient

### สูตร Distribution Focal Loss

$$
 DFL(S_i, y) = -(y_{i+1} - y)\log S_i - (y - y_i)\log S_{i+1}
$$

### Gradient w.r.t. $S_i$ (softmax output)

$$
 \frac{\partial DFL}{\partial S_i} = -\frac{y_{i+1} - y}{S_i}
$$

**นัยยะ:**

- Gradient เป็นสัดส่วนกับ proximity ของ target $y$ ถึง bin boundaries
- ถ้า $y$ อยู่ใกล้ $y_i$ → gradient ของ $S_i$ สูง
- **Distribution learning ทำให้ model ไม่ต้อง regress ค่าแม่นเป๊ะ** แต่กระจาย probability ไปหลาย bins

## G.5 Gradient Flow Analysis in YOLO Backbones

### Plain Sequential (v1)

$$
 \frac{\partial L}{\partial x_0} = \prod_{i=0}^{L} \frac{\partial f_i}{\partial x_i}
$$

**ปัญหา:** vanishing gradient เมื่อ $L$ มาก

### Residual (v3+)

$$
 \frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}\mathcal{F}_i\right)
$$

**ข้อดี:** gradient มี "1" term → ไม่ vanish

### CSP (v4+)

$$
 \frac{\partial L}{\partial x} = \lambda_1 \frac{\partial L}{\partial x_{path1}} + \lambda_2 \frac{\partial L}{\partial x_{path2}}
$$

**ข้อดี:** gradient flow ถูกแบ่งเป็น 2 paths → ลด redundancy

## G.6 ตารางสรุป Gradient Properties

| Backbone      | Gradient Path   | Vanishing Risk | Training Stability |
| :------------ | :-------------- | :------------- | :----------------- |
| Plain (v1)    | Single chain    | สูง            | ต่ำ                |
| Residual (v3) | Skip + main     | กลาง           | ดี                 |
| CSP (v4--v5)  | Split + merge   | ต่ำ            | ดีมาก              |
| ELAN (v7)     | Multi-branch    | ต่ำมาก         | ดีเยี่ยม           |
| C2f (v8)      | Split + partial | ต่ำ            | ดีมาก              |
