# ภาคผนวก F --- Dataset & Benchmark Reference

> ตาราง benchmark มาตรฐานและผลเปรียบเทียบ YOLO ข้ามเวอร์ชัน

## F.1 COCO Dataset Overview

| รายการ      | รายละเอียด                                 |
| :---------- | :----------------------------------------- |
| ชื่อ        | Microsoft COCO (Common Objects in Context) |
| Version     | 2017                                       |
| Train       | 118,287 images                             |
| Val         | 5,000 images                               |
| Test-dev    | 20,288 images                              |
| Categories  | 80 classes                                 |
| Metric หลัก | AP@0.5:0.95 (primary), AP@0.5, AP@0.75     |

### Evaluation Metrics

$$
 AP = \int_0^1 p(r)\,dr
$$

- $p(r)$: precision ที่ recall level $r$
- **AP@0.5:** IoU threshold = 0.5 (lenient)
- **AP@0.5:0.95:** เฉลี่ย AP จาก IoU 0.5 ถึง 0.95 ทุก 0.05 (strict)

## F.2 COCO Benchmark --- YOLO ข้ามเวอร์ชัน

### Small Models (< 15M params)

| Model       | Input | Params (M) | FLOPs (G) | AP@0.5:0.95 | AP@0.5 | FPS (T4) |
| :---------- | :---- | :--------- | :-------- | :---------- | :----- | :------- |
| YOLOv5s     | 640   | 7.2        | 16.5      | 37.4        | 56.8   | 130      |
| YOLOv7-tiny | 640   | 6.2        | 13.7      | 37.4        | 55.2   | 149      |
| YOLOv8s     | 640   | 11.2       | 28.6      | 44.9        | 61.8   | 128      |
| YOLOX-S     | 640   | 9.0        | 26.8      | 40.5        | ---    | 102      |
| YOLOv11s    | 640   | 9.4        | 21.5      | 47.0        | 63.0   | 120      |

### Medium Models (15--50M params)

| Model    | Input | Params (M) | FLOPs (G) | AP@0.5:0.95 | AP@0.5 | FPS (T4) |
| :------- | :---- | :--------- | :-------- | :---------- | :----- | :------- |
| YOLOv5m  | 640   | 21.2       | 49.0      | 45.4        | 64.1   | 80       |
| YOLOv8m  | 640   | 25.9       | 78.9      | 50.2        | 67.2   | 78       |
| YOLOv7   | 640   | 36.9       | 104.7     | 51.4        | 69.7   | 70       |
| YOLOv9-C | 640   | 25.3       | 102.1     | 53.0        | 70.2   | 62       |
| YOLOv11m | 640   | 20.1       | 68.0      | 51.5        | 68.5   | 75       |

### Large Models (50M+ params)

| Model      | Input | Params (M) | FLOPs (G) | AP@0.5:0.95 | AP@0.5 | FPS (T4) |
| :--------- | :---- | :--------- | :-------- | :---------- | :----- | :------- |
| YOLOv5x    | 640   | 86.7       | 205.7     | 50.7        | 68.9   | 33       |
| YOLOv8x    | 640   | 68.2       | 257.8     | 53.9        | 71.0   | 38       |
| YOLOv7-E6E | 1280  | 151.7      | 843.2     | 56.8        | 74.4   | 18       |
| YOLOv9-E   | 640   | 57.3       | 189.0     | 55.6        | 72.8   | 28       |

## F.3 AP Across Object Sizes

| Model    | AP_S | AP_M | AP_L |
| :------- | :--- | :--- | :--- |
| YOLOv5s  | 21.7 | 41.6 | 49.3 |
| YOLOv8s  | 27.7 | 49.5 | 59.3 |
| YOLOv8m  | 33.6 | 55.4 | 65.0 |
| YOLOv9-E | 37.9 | 60.1 | 71.4 |

> **Key Insight:** Small object detection ($AP_S$) ยังคงเป็นจุดอ่อนหลักของทุก YOLO --- ต้องการ higher resolution features

## F.4 Inference Platforms

| Platform          | GPU    | Memory | ใช้ใน benchmark     |
| :---------------- | :----- | :----- | :------------------ |
| NVIDIA T4         | Turing | 16 GB  | Ultralytics default |
| NVIDIA V100       | Volta  | 32 GB  | academic standard   |
| NVIDIA A100       | Ampere | 80 GB  | large models        |
| NVIDIA Jetson AGX | Orin   | 32 GB  | edge deployment     |

## F.5 Dataset อื่นที่มักใช้ในงานวิจัย YOLO

| Dataset        | Images | Classes | ลักษณะ                  |
| :------------- | :----- | :------ | :---------------------- |
| PASCAL VOC     | 11,540 | 20      | legacy benchmark        |
| Objects365     | 2M     | 365     | large-scale pretraining |
| Open Images v7 | 9M     | 600     | Google's large-scale    |
| LVIS v1.0      | 164k   | 1,203   | long-tail distribution  |
| BDD100k        | 100k   | 10      | autonomous driving      |
| Roboflow 100   | varied | varied  | domain-specific tasks   |
