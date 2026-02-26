# ภาคผนวก A --- ตารางสัญลักษณ์

> สัญลักษณ์ทั้งหมดที่ใช้ตลอดเล่ม จัดตามหมวดหมู่เพื่อความสะดวกในการอ้างอิง

## A.1 สัญลักษณ์ทั่วไป

| สัญลักษณ์    | ความหมาย                         | หน่วย/ขอบเขต                           |
| :----------- | :------------------------------- | :------------------------------------- |
| $\mathbf{I}$ | ภาพอินพุต                        | $\in \mathbb{R}^{H \times W \times 3}$ |
| $H, W$       | ความสูง, ความกว้างของภาพ         | pixels                                 |
| $B$          | batch size                       | integer                                |
| $C$          | จำนวนคลาส                        | integer                                |
| $S$          | ขนาด grid (ต่อแกน)               | integer                                |
| $D$          | ความลึกของ prediction tensor     | integer                                |
| $\theta$     | พารามิเตอร์ของเครือข่าย          | $\in \mathbb{R}^n$                     |
| $f_\theta$   | ฟังก์ชันแม็ปจากภาพสู่ prediction | ---                                    |

## A.2 Bounding Box

| สัญลักษณ์              | ความหมาย                                         |
| :--------------------- | :----------------------------------------------- |
| $b = (x, y, w, h)$     | bounding box coordinates                         |
| $(t_x, t_y, t_w, t_h)$ | predicted offsets (raw)                          |
| $(c_x, c_y)$           | grid cell top-left index                         |
| $(p_w, p_h)$           | anchor prior dimensions                          |
| $(b_x, b_y, b_w, b_h)$ | decoded bounding box                             |
| $(l, t, r, b)$         | anchor-free distances (left, top, right, bottom) |

## A.3 Score & Classification

| สัญลักษณ์             | ความหมาย                         |
| :-------------------- | :------------------------------- |
| $s_i \in [0,1]$       | confidence score ของวัตถุที่ $i$ |
| $c_i \in \mathcal{C}$ | คลาสของวัตถุที่ $i$              |
| $P_{obj}$             | objectness probability           |
| $P_{class_i}$         | probability ของคลาส $i$          |
| $\sigma(\cdot)$       | sigmoid function                 |
| $N$                   | จำนวนวัตถุในภาพ                  |

## A.4 Feature Maps

| สัญลักษณ์                                        | ความหมาย                                  |
| :----------------------------------------------- | :---------------------------------------- |
| $F_l \in \mathbb{R}^{C_l \times H_l \times W_l}$ | feature map ระดับ $l$                     |
| $\phi_l$                                         | transformation function ระดับ $l$         |
| $P_3, P_4, P_5$                                  | feature pyramid levels (stride 8, 16, 32) |
| $C_l$                                            | จำนวน channels ระดับ $l$                  |
| $A$                                              | จำนวน anchors ต่อ grid cell               |

## A.5 Loss Functions

| สัญลักษณ์                           | ความหมาย                           |
| :---------------------------------- | :--------------------------------- |
| $L_{box}$                           | localization loss                  |
| $L_{cls}$                           | classification loss                |
| $L_{obj}$                           | objectness loss                    |
| $\text{IoU}$                        | Intersection over Union            |
| $\text{GIoU}$                       | Generalized IoU                    |
| $\text{CIoU}$                       | Complete IoU                       |
| $\rho(\mathbf{b}, \mathbf{b}_{gt})$ | Euclidean distance ระหว่าง centers |
| $v$                                 | aspect ratio consistency term      |
| $\alpha$                            | tradeoff coefficient               |

## A.6 Dynamic Assignment

| สัญลักษณ์                    | ความหมาย                         |
| :--------------------------- | :------------------------------- |
| $t = s^\alpha \cdot u^\beta$ | alignment score                  |
| $s$                          | classification score             |
| $u$                          | IoU score                        |
| $\alpha, \beta$              | hyperparameters สำหรับ alignment |

## A.7 สัญลักษณ์เฉพาะบท

| สัญลักษณ์           | บท  | ความหมาย                |
| :------------------ | :-- | :---------------------- |
| $\mathcal{A}$       | 1   | เซตของ annotations      |
| $\mathcal{Y}$       | 1   | structured output space |
| $\mathcal{U}$       | 2   | upsampling operator     |
| $\mathcal{H}$       | 21  | entropy function        |
| $\Omega(\theta)$    | 4   | regularization term     |
| $\mathcal{L}_{ERM}$ | 4   | empirical risk          |
