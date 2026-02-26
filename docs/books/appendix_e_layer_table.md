# ภาคผนวก E --- Layer Configuration Table

> ตารางโครงสร้าง layer-by-layer สำหรับ YOLO แต่ละเวอร์ชัน

## E.1 YOLOv3 (Darknet-53 Backbone)

| ชั้น   | Type          | Filters  | Size/Stride | Output  | หมายเหตุ      |
| :----- | :------------ | :------- | :---------- | :------ | :------------ |
| 0      | Conv+BN+Leaky | 32       | 3×3/1       | 416×416 | initial conv  |
| 1      | Conv+BN+Leaky | 64       | 3×3/2       | 208×208 | downsample    |
| 2--3   | Residual ×1   | 32/64    | 1×1, 3×3    | 208×208 |               |
| 4      | Conv+BN+Leaky | 128      | 3×3/2       | 104×104 | downsample    |
| 5--8   | Residual ×2   | 64/128   | 1×1, 3×3    | 104×104 |               |
| 9      | Conv+BN+Leaky | 256      | 3×3/2       | 52×52   | downsample    |
| 10--25 | Residual ×8   | 128/256  | 1×1, 3×3    | 52×52   | **P3 output** |
| 26     | Conv+BN+Leaky | 512      | 3×3/2       | 26×26   | downsample    |
| 27--42 | Residual ×8   | 256/512  | 1×1, 3×3    | 26×26   | **P4 output** |
| 43     | Conv+BN+Leaky | 1024     | 3×3/2       | 13×13   | downsample    |
| 44--51 | Residual ×4   | 512/1024 | 1×1, 3×3    | 13×13   | **P5 output** |

## E.2 YOLOv5s

| Component | Module          | Depth     | Width | Output   |
| :-------- | :-------------- | :-------- | :---- | :------- |
| Backbone  | Focus           | 1         | 32    | 320×320  |
|           | Conv            | 1         | 64    | 160×160  |
|           | C3              | 1         | 128   | 80×80    |
|           | Conv            | 1         | 256   | 40×40    |
|           | C3              | 3         | 256   | 40×40    |
|           | Conv            | 1         | 512   | 20×20    |
|           | C3              | 1         | 512   | 20×20    |
|           | SPPF            | 1         | 512   | 20×20    |
| Neck      | Upsample+Concat | ---       | ---   | 40×40    |
|           | C3              | 1         | 256   | 40×40    |
|           | Upsample+Concat | ---       | ---   | 80×80    |
|           | C3              | 1         | 128   | 80×80    |
|           | Conv            | 1         | 128   | 40×40    |
|           | C3              | 1         | 256   | 40×40    |
|           | Conv            | 1         | 256   | 20×20    |
|           | C3              | 1         | 512   | 20×20    |
| Head      | Detect          | 3 outputs | ---   | P3/P4/P5 |

## E.3 YOLOv8n

| Component | Module    | Channels | Output  | Notes         |
| :-------- | :-------- | :------- | :------ | :------------ |
| Backbone  | Conv      | 16       | 320×320 | stem          |
|           | Conv      | 32       | 160×160 | stride 2      |
|           | C2f       | 32       | 160×160 | 1 bottleneck  |
|           | Conv      | 64       | 80×80   | stride 2      |
|           | C2f       | 64       | 80×80   | 2 bottlenecks |
|           | Conv      | 128      | 40×40   | stride 2      |
|           | C2f       | 128      | 40×40   | 2 bottlenecks |
|           | Conv      | 256      | 20×20   | stride 2      |
|           | C2f       | 256      | 20×20   | 1 bottleneck  |
|           | SPPF      | 256      | 20×20   |               |
| Neck      | Upsample  | ---      | 40×40   | top-down      |
|           | C2f       | 128      | 40×40   | fusion        |
|           | Upsample  | ---      | 80×80   | top-down      |
|           | C2f       | 64       | 80×80   | fusion        |
|           | Conv      | 64       | 40×40   | bottom-up     |
|           | C2f       | 128      | 40×40   | fusion        |
|           | Conv      | 128      | 20×20   | bottom-up     |
|           | C2f       | 256      | 20×20   | fusion        |
| Head      | Decoupled | 3 scales | ---     | anchor-free   |

## E.4 Model Scale Comparison

| Scale | Width Mult. | Depth Mult. | Params (M) | FLOPs (G) |
| :---- | :---------- | :---------- | :--------- | :-------- |
| v8n   | 0.25        | 0.33        | 3.2        | 8.7       |
| v8s   | 0.50        | 0.33        | 11.2       | 28.6      |
| v8m   | 0.75        | 0.67        | 25.9       | 78.9      |
| v8l   | 1.00        | 1.00        | 43.7       | 165.2     |
| v8x   | 1.25        | 1.00        | 68.2       | 257.8     |
