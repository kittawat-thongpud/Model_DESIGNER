## ความเหมาะสม: Anchor-based vs Query-based (N) สำหรับงานความละเอียด 1080p (เช่น IDD)

### 1) มิติของปัญหา (1080p)

ภาพ 1920×1080 มี spatial tokens สูง
หาก stride = 8:

[
H \times W \approx 240 \times 135 \approx 32{,}400
]

* **Anchor-based** → hypotheses = (H \times W \times A)
  ถ้า A=3 → ~97k boxes
* **Query-based** → hypotheses = (N) (เช่น 300)

---

# A) Anchor-based (YOLO-style)

### เหมาะเมื่อ

* ต้องการ **real-time บน edge/GPU เล็ก**
* scene มีวัตถุหนาแน่นมาก
* latency สำคัญกว่า global reasoning

### จุดแข็ง

* Complexity ใกล้ (O(HW))
* Conv-friendly
* Throughput สูง

### ข้อจำกัดใน 1080p

* Anchor redundancy สูงมาก
* NMS cost เพิ่มตามจำนวน box
* Long-range occlusion reasoning จำกัด

### สรุปความเหมาะสม

> เหมาะกับระบบ production latency-critical
> แต่ไม่ optimal สำหรับ scene ซับซ้อนระดับเมือง

---

# B) Query-based (DETR / RF-DETR)

### เหมาะเมื่อ

* Scene ซับซ้อน (traffic, occlusion, scale diversity)
* ต้องการ **one-to-one matching**
* ต้องการ eliminate NMS

### จุดแข็ง

* Hypothesis คงที่ (N)
* Global context modeling
* No duplicate suppression

### ความท้าทายใน 1080p

Encoder attention:

[
O((HW)^2)
]

ถ้าไม่ลด resolution → memory สูง

### วิธีแก้

* Downsample ก่อนเข้า transformer
* Hybrid CNN encoder
* Deformable attention

---

# C) เปรียบเทียบเชิงความเหมาะสม

| ปัจจัย        | Anchor-based | Query-based           |
| ------------- | ------------ | --------------------- |
| 1080p memory  | ดี           | ต้อง optimize         |
| Dense objects | ดีมาก        | จำกัดด้วย N           |
| Occlusion     | จำกัด        | ดีมาก                 |
| Latency       | ต่ำ          | สูงกว่า               |
| Post-process  | ต้อง NMS     | ไม่ต้อง               |
| Stability     | สูง          | ต้อง careful training |

---

# D) สำหรับ IDD (urban driving, 1080p)

ลักษณะข้อมูล:

* Small objects (sign, pedestrian)
* Large objects (bus, truck)
* Occlusion
* Long-range dependency

### Recommendation

## ถ้าเน้น Real-time:

**Anchor-free YOLO (decoupled head + TAL + PAFPN)**
ดีที่สุดเชิง latency / accuracy balance

---

## ถ้าเน้น Accuracy สูงสุด:

**Deformable DETR / RF-DETR style hybrid**

เหตุผล:

* Urban scene ต้อง global reasoning
* NMS removal ลด error propagation
* One-to-one matching เหมาะกับ traffic scene

---

# E) ค่า N ที่เหมาะสมสำหรับ 1080p

IDD เฉลี่ย ~10–30 objects ต่อภาพ

แนะนำ:

[
N = 300
]

เผื่อ worst-case crowded scene

---

# F) บทสรุปความเหมาะสมเชิงทฤษฎี

Anchor-based:
[
\text{Entropy reduction} \rightarrow local
]

Query-based:
[
\text{Entropy reduction} \rightarrow global
]

ใน urban scene entropy สูง →
**Global reasoning ได้เปรียบ**

---

# Final Verdict

| เป้าหมาย                | เหมาะสม                           |
| ----------------------- | --------------------------------- |
| Edge deployment         | YOLO anchor-free                  |
| Research-grade accuracy | RF-DETR                           |
| Balanced hybrid         | CNN backbone + Deformable decoder |

---

หากต้องการต่อ:

* คำนวณ FLOPs ที่ 1080p จริง
* เปรียบเทียบ memory footprint
* ออกแบบ hybrid ที่เหมาะกับ IDD โดยเฉพาะ
* วิเคราะห์เชิง entropy-bound ระหว่างสองแบบ

ระบุแนวที่ต้องการต่อได้ทันที.
