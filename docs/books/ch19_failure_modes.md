# บทที่ 19 --- Failure Modes และ Domain Limitations

## 19.1 Small-Object Collapse

### ปัญหา

วัตถุ $<32 \times 32$ pixels บน **COCO** มี mAP ต่ำกว่า large objects อย่างมาก:

| Object Size           | Coverage          | mAP@0.5:0.95 (ตัวอย่าง v8s) | ผลต่าง   |
| :-------------------- | :---------------- | :-------------------------- | :------- |
| Small ($<32^2$)       | 41% ของ instances | ~25%                        | Baseline |
| Medium ($32^2--96^2$) | 34%               | ~50%                        | +25%     |
| Large ($>96^2$)       | 25%               | ~60%                        | +35%     |

### สาเหตุเชิงโครงสร้าง

1.  **Resolution loss**: P3 (stride 8) → วัตถุ $8 \times 8$ px = 1 pixel ใน feature map
2.  **Limited features**: 1 pixel ไม่มี spatial pattern → classification ยาก
3.  **Assignment conflict**: หลายวัตถุเล็กอยู่ใน grid cell เดียว

## 19.2 Domain Shift Sensitivity

### ปัญหา

Model ที่ train บน COCO (everyday objects, well-lit, varied angles) ทำงานไม่ดีเมื่อ deploy บน domain อื่น

| Source Domain | Target Domain         | Performance Drop | สาเหตุ                      |
| :------------ | :-------------------- | :--------------- | :-------------------------- |
| COCO          | Medical imaging       | ≈ 30--50%        | Texture, scale ต่างมาก      |
| COCO          | Aerial/satellite      | ≈ 20--40%        | Top-down view, tiny objects |
| COCO          | Night/infrared        | ≈ 15--30%        | Lighting distribution shift |
| COCO          | Industrial inspection | ≈ 20--35%        | Rare defect classes         |

### สาเหตุเชิงทฤษฎี

จาก statistical learning view:

$$
 R_{target}(\theta) = R_{source}(\theta) + d(\mathcal{D}_{source}, \mathcal{D}_{target})
$$

โดยที่ $d(\cdot, \cdot)$ เป็น distribution divergence --- ยิ่ง domain ต่างกันมาก risk เพิ่มขึ้น

## 19.3 Label Noise Tolerance

### ปัญหา

Annotation errors ใน training data:

| ประเภท Noise        | ตัวอย่าง                                   | ผลกระทบ                  |
| :------------------ | :----------------------------------------- | :----------------------- |
| Missing annotations | GT box ถูกลืม → false negative ใน training | Model learns to suppress |
| Inaccurate boxes    | Box ไม่ tight → regression target มี noise | IoU loss degraded        |
| Wrong class labels  | Cat labeled as dog                         | Cls loss ผิดทิศ          |

### ผลต่อ Label Assignment

| Assignment | Noise Tolerance | เหตุผล                                |
| :--------- | :-------------- | :------------------------------------ |
| Static IoU | ⚠️ ปานกลาง      | IoU with noisy box = noisy assignment |
| SimOTA     | ✅ ดีกว่า       | Dynamic adaptation                    |
| TAL        | ✅ ดี           | Task-aligned weighting                |
| One-to-one | ❌ **ไม่ดี**    | ดู Hypothesis 19.1                    |

## 19.4 Dense Scene Failure

### ปัญหา

เมื่อมีวัตถุจำนวนมากอยู่ใกล้กัน (dense scenes):

1.  **Slot collision** --- หลาย objects ใน 1 grid cell → prediction capacity ไม่พอ
2.  **NMS sensitivity** --- overlapping objects ถูก suppress ผิด
3.  **Assignment breakdown** --- มาก GT, น้อย positive samples → gradient imbalance

### ตัวอย่าง

| Scenario             | Object Count | Grid Density | Performance |
| :------------------- | :----------- | :----------- | :---------- |
| Single object        | 1            | ต่ำ          | ✅ ดีมาก    |
| Normal (COCO avg ~7) | 7            | ปานกลาง      | ✅ ดี       |
| Dense (crowd)        | >50          | สูงมาก       | ⚠️ ลดลง     |
| Extreme dense        | >200         | สูงสุด       | ❌ collapse |

## 19.5 Hypothesis 19.1: One-to-One Assignment Under Ambiguity

> **Hypothesis 19.1** (One-to-One Assignment Degrades Under Annotation Ambiguity)
>
> ระบบ one-to-one assignment (YOLOv10) มี performance drop มากกว่า one-to-many เมื่อ annotation quality ต่ำ
>
> **Assumptions:**
>
> 1.  Annotation ambiguity = บาง GT boxes มี IoU ต่ำกับ correct location (noisy annotations)
> 2.  One-to-one assigns เพียง 1 prediction ต่อ GT
> 3.  เปรียบเทียบบน dataset เดียวกัน, สภาวะ training เดียวกัน
>
> **Reasoning:**
>
> ใน one-to-many:
>
> - หลาย predictions ถูก assign ให้ 1 GT → ถ้า GT noisy, loss ถูก "เฉลี่ย" (smoothed) ข้ามหลาย predictions
> - Gradient diversity สูง → robust ต่อ individual label noise
>
> ใน one-to-one:
>
> - เพียง 1 prediction ถูก assign → **ถ้า GT noisy, ทั้ง gradient เป็นไปในทิศ noisy**
> - ไม่มีการ "เฉลี่ย" → sensitive ต่อ single annotation error
>
> **Expected signature**: mAP drop ของ one-to-one > mAP drop ของ one-to-many เมื่อเพิ่ม label noise rate
>
> **Status**: ยังไม่มี systematic empirical validation $\square$

## 19.6 Mitigation Strategies

| Failure Mode          | Mitigation                                             | ดูบท       |
| :-------------------- | :----------------------------------------------------- | :--------- |
| Small-object collapse | เพิ่ม P2 scale (stride 4), mosaic augmentation         | Ch.2, Ch.8 |
| Domain shift          | Fine-tuning, domain adaptation, test-time augmentation | Ch.4       |
| Label noise           | Soft labels, TAL (noise-tolerant), data cleaning       | Ch.10      |
| Dense scenes          | เพิ่ม anchor slots, NMS tuning, one-to-many assignment | Ch.10      |

## เอกสารอ้างอิง

1.  Kisantal, M., et al. (2019). "Augmentation for small object detection." _CVPRW 2019_. arXiv:1902.07296
2.  Oksuz, K., et al. (2020). "Imbalance Problems in Object Detection: A Review." _IEEE TPAMI_. arXiv:1909.00169
3.  Wang, A., et al. (2024). "YOLOv10." arXiv:2405.14458
