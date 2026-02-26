# Model Parameters & FLOPs Display

## สรุปการเพิ่มฟีเจอร์

เพิ่มการแสดง **Parameters** และ **FLOPs** ของโมเดลใน metadata และระหว่าง training

## การทำงาน

### 1. คำนวณ Params & FLOPs

**Endpoint**: `POST /api/models/{model_id}/validate`

```python
# Calculate parameters
n_params = sum(p.numel() for p in model.model.parameters())

# Calculate FLOPs from Ultralytics
info_dict = model.model.info(verbose=False, imgsz=640)
flops = info_dict.get('GFLOPs')  # GigaFLOPs
```

**Response**:
```json
{
  "valid": true,
  "params": 3157200,
  "flops": 8.9,
  "layers": 225,
  "message": "Valid — 3,157,200 params, 225 layers, 8.9 GFLOPs"
}
```

### 2. บันทึกใน Model Metadata

**Backend**: `model_storage.py`
```python
results.append({
    "model_id": rec["model_id"],
    "name": rec["name"],
    "params": rec.get("params"),      # จำนวน parameters
    "flops": rec.get("flops"),        # GigaFLOPs
    ...
})
```

**Schema**: `ModelSummary`
```python
class ModelSummary(BaseModel):
    model_id: str
    name: str
    params: int | None = None
    flops: float | None = None
    ...
```

### 3. แสดงใน CreateTrainJobModal

**Location**: Model Configuration Header

**Display**:
```
Configuration: YOLOv8-Detection
├─ Task: detect
├─ ID: abc12345
├─ Input: [3, 640, 640]
├─ Params: 3.16M          ← ใหม่ (สีม่วง)
└─ FLOPs: 8.9 G           ← ใหม่ (สีเหลือง)
```

**Format**:
- Params: `(params / 1e6).toFixed(2)M` → "3.16M"
- FLOPs: `flops.toFixed(1) G` → "8.9 G"

### 4. แสดงใน JobConfiguration

**Location**: System Column → Model Info Section

**Display**:
```
System
├─ Workers: 8
├─ AMP: true
├─ Device: Auto
├─ Seed: 0
├─ Model Scale: N
└─ Model Info
   ├─ Layers: 225
   ├─ Params: 3.16M       ← highlighted
   └─ FLOPs: 8.9 G        ← highlighted
```

## ไฟล์ที่แก้ไข

### Backend (4 files)
1. `backend/app/controllers/model_controller.py`
   - เพิ่มการคำนวณ FLOPs ใน `validate_model()`
   - Return `flops` ใน response

2. `backend/app/services/model_storage.py`
   - เพิ่ม `params` และ `flops` ใน `list_models()`

3. `backend/app/schemas/model.py`
   - เพิ่ม `params` และ `flops` ใน `ModelSummary`

### Frontend (3 files)
1. `frontend/src/types/index.ts`
   - เพิ่ม `params` และ `flops` ใน `ModelSummary`
   - เพิ่ม `model_flops` ใน `JobRecord`

2. `frontend/src/components/CreateTrainJobModal.tsx`
   - แสดง params และ FLOPs ใน header

3. `frontend/src/components/JobConfiguration.tsx`
   - เพิ่ม Model Info section
   - แสดง layers, params, FLOPs

## การใช้งาน

### 1. Validate Model เพื่อคำนวณ Params & FLOPs

```bash
POST /api/models/{model_id}/validate
```

Response จะมี `params` และ `flops` ที่คำนวณได้

### 2. บันทึกค่าใน Model Metadata

ค่า params และ FLOPs จะถูกบันทึกใน `record.json`:
```json
{
  "model_id": "abc123",
  "name": "YOLOv8n",
  "params": 3157200,
  "flops": 8.9,
  ...
}
```

### 3. ดูข้อมูลตอนสร้าง Training Job

เมื่อเลือก model ใน CreateTrainJobModal จะเห็น:
- **Params**: 3.16M (สีม่วง)
- **FLOPs**: 8.9 G (สีเหลือง)

### 4. ดูข้อมูลใน Job Details

ใน JobConfiguration → System → Model Info:
- Layers: 225
- Params: 3.16M (highlighted)
- FLOPs: 8.9 G (highlighted)

## ประโยชน์

### 1. เปรียบเทียบ Model Complexity
- เห็นว่า model ไหนใหญ่กว่า (params)
- เห็นว่า model ไหนใช้ computation มากกว่า (FLOPs)

### 2. เลือก Model ที่เหมาะสม
- Model เล็ก (น้อย params) → เร็ว, ใช้ memory น้อย
- Model ใหญ่ (มาก params) → แม่นยำกว่า แต่ช้ากว่า

### 3. Track Model Efficiency
- FLOPs/Param ratio → efficiency
- เปรียบเทียบ architectures

### 4. Resource Planning
- รู้ว่าต้องใช้ GPU memory เท่าไหร่
- ประมาณ inference time

## ตัวอย่างค่า

### YOLOv8 Models
| Scale | Params | FLOPs |
|-------|--------|-------|
| n     | 3.2M   | 8.7G  |
| s     | 11.2M  | 28.6G |
| m     | 25.9M  | 78.9G |
| l     | 43.7M  | 165.2G|
| x     | 68.2M  | 257.8G|

### Classification Models
| Model | Params | FLOPs |
|-------|--------|-------|
| ResNet18 | 11.7M | 1.8G |
| ResNet50 | 25.6M | 4.1G |
| EfficientNet-B0 | 5.3M | 0.4G |

## Next Steps (Optional)

1. **Auto-calculate on save**: คำนวณ params/FLOPs อัตโนมัติตอน save model
2. **Model comparison**: เปรียบเทียบหลาย models พร้อมกัน
3. **Efficiency metrics**: แสดง FLOPs/Param, Params/Accuracy ratios
4. **Memory estimation**: ประมาณ GPU memory usage จาก params
5. **Inference speed**: ประมาณ inference time จาก FLOPs
