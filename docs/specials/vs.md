# 1) Anchor-based คืออะไร?

**Anchor-based detection** คือการกำหนด “กล่องตั้งต้น” (prior boxes) ไว้ล่วงหน้าในแต่ละ grid cell
โมเดลไม่ทำนาย box จากศูนย์ แต่ทำนาย **offset** จาก anchor

---

## โครงสร้างพื้นฐาน

สมมติ feature map:

[
F \in \mathbb{R}^{B \times C \times H \times W}
]

แต่ละตำแหน่ง (i,j) มี **A anchors**

จำนวน hypothesis ทั้งหมด:

[
H \times W \times A
]

เช่น:

* 80 × 80 × 3 = 19,200 anchors

---

## Anchor ทำงานยังไง?

แต่ละ anchor มี prior:

[
a = (x_a, y_a, w_a, h_a)
]

โมเดลทำนาย:

[
(t_x, t_y, t_w, t_h)
]

แล้ว decode เป็น box จริง:

[
x = x_a + t_x w_a
]
[
y = y_a + t_y h_a
]
[
w = w_a e^{t_w}
]
[
h = h_a e^{t_h}
]

---

## ข้อดี

* ช่วย stabilize training
* ลด search space

## ปัญหา

* Hyperparameters เยอะ
* Prior mismatch → entropy สูง
* Redundant boxes มาก → ต้อง NMS

---

# 2) แล้ว N คืออะไรใน DETR / RF-DETR?

N = จำนวน **object queries**

ไม่ใช่ anchor
ไม่ใช่ grid
ไม่ผูก spatial location

---

## รูปแบบข้อมูล

Backbone output:

[
[B, C, H, W]
]

flatten:

[
[B, HW, C]
]

สร้าง learnable queries:

[
Q \in \mathbb{R}^{B \times N \times C}
]

เช่น:

[
N = 100 \text{ หรือ } 300
]

---

# 3) N ทำงานยังไง?

แต่ละ query คือ:

> “ตัวแทน 1 วัตถุที่อาจมีอยู่”

---

## ขั้นตอน Decoder

### Step 1: Query self-attention

queries คุยกันเอง
ลด duplication

---

### Step 2: Cross-attention

[
\text{CrossAttn}(Q, F_{enc})
]

* Query = object hypothesis
* Key/Value = spatial tokens

query จะไป “โฟกัส” pixel ที่เกี่ยวข้อง

---

## Output

[
[B, N, 4+nc]
]

แต่ละ query = 1 object prediction

---

# 4) เปรียบเทียบ Anchor vs N

|                  | Anchor-based   | Query-based (N) |
| ---------------- | -------------- | --------------- |
| หน่วยพื้นฐาน     | Grid + anchor  | Object query    |
| จำนวน hypothesis | H×W×A (หมื่น+) | N (100–300)     |
| ผูก spatial?     | ใช่            | ไม่             |
| Assignment       | many-to-one    | one-to-one      |
| NMS              | ต้องใช้        | ไม่ต้อง         |
| Redundancy       | สูง            | ต่ำ             |

---

# 5) Intuition เชิงภาพ

## Anchor-based (YOLO)

> ทุกตำแหน่งในภาพลองทำนายวัตถุ

เหมือนเอาตาข่ายคลุมภาพ

---

## N queries (DETR)

> ส่ง N นักสำรวจไปค้นหาวัตถุ

แต่ละคนต้องรับผิดชอบวัตถุเดียว

---

# 6) แล้ว N เลือกยังไง?

N ต้องมากกว่า:

[
\max(\text{objects per image})
]

เช่น:

* COCO เฉลี่ย ~7 objects
* เลือก N = 100 หรือ 300

เหตุผล:

* เผื่อกรณี dense scene
* เหลือ unused queries ได้ (predict background)

---

# 7) Complexity เปรียบเทียบ

## YOLO

[
O(HW)
]

## DETR

Self-attention:

[
O((HW)^2)
]

Decoder:

[
O(NHW)
]

ถ้า:

* HW = 2000
* N = 300

Decoder cost ≈ 600k
Encoder cost ≈ 4M

ยัง manageable

---

# 8) สรุปแบบโครงสร้าง

Anchor-based:

* Prior spatial
* Local regression
* External suppression

Query-based:

* Prior semantic
* Global attention
* Internal matching

---

ถ้าต้องการต่อ ผมสามารถ:

* วิเคราะห์ว่า 1080p ควรเลือก N เท่าไร
* ออกแบบ hybrid anchor + query
* เปรียบเทียบ entropy reduction เชิงทฤษฎี
* วิเคราะห์ memory / FLOPs เชิงตัวเลข

เลือกแนวต่อได้เลย.
