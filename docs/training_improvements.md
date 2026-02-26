# Training System Improvements

## สรุปการแก้ไข 3 ข้อ

### 1. ✅ เพิ่ม Scale Selector ใน Job Creation

**Frontend**: `CreateTrainJobModal.tsx`
- เพิ่ม state `modelScale` (default: 'n')
- เพิ่ม UI selector สำหรับเลือก scale: n, s, m, l, x
- ส่ง `model_scale` ไปยัง backend ใน `TrainRequest`

**TypeScript Types**: `types/index.ts`
- เพิ่ม `model_scale?: string` ใน `TrainRequest`
- เพิ่ม `model_scale?: string` ใน `JobRecord`

**UI Features**:
```tsx
<div className="flex gap-2">
  {['n', 's', 'm', 'l', 'x'].map(scale => (
    <button onClick={() => setModelScale(scale)}>
      {scale.toUpperCase()}
    </button>
  ))}
</div>
```

### 2. ✅ แสดง Partition Configuration ใน Job Config

**Frontend**: `JobConfiguration.tsx`
- เพิ่ม props `partitions` และ `modelScale`
- แสดง partition list พร้อม splits ที่เลือก (train/val/test)
- แสดง model scale ใน System column

**Display Format**:
```
Partitions (2)
├─ partition_0
│  └─ train + val
└─ partition_1
   └─ train + val + test
```

**JobDetailPage.tsx**:
```tsx
<JobConfiguration 
  config={job.config} 
  partitions={job.partitions}
  modelScale={job.model_scale}
/>
```

### 3. ✅ Weight Storage จำแนก Scale

**Backend**: `weight_storage.py`
- เพิ่ม parameter `model_scale` ใน `save_weight_meta()`
- บันทึก `model_scale` ใน weight metadata
- Default เป็น "n" ถ้าไม่ระบุ

**Backend**: `ultra_trainer.py`
- ส่ง `model_scale` จาก job ไปยัง `save_weight_meta()`
- Weight metadata จะมี field `model_scale` สำหรับแยกแยะ

**Weight Metadata Structure**:
```json
{
  "weight_id": "abc123",
  "model_id": "xyz789",
  "model_name": "YOLOv8",
  "model_scale": "n",
  "dataset": "coco8",
  "epochs_trained": 100,
  ...
}
```

## ประโยชน์

### Scale Selector
- ผู้ใช้เลือก scale ได้ง่าย (n/s/m/l/x)
- ไม่ต้องแก้ YAML manually
- แสดงคำอธิบาย scale ใน UI

### Partition Display
- เห็นว่าใช้ partition ไหนบ้างใน training
- เห็น splits ที่เลือก (train/val/test)
- ไม่ต้องเปิด dataset page เพื่อดู

### Weight Scale Tracking
- แยกแยะ weights ตาม scale ได้
- รู้ว่า weight ไหนมาจาก scale อะไร
- สามารถ filter/search weights by scale ได้ในอนาคต

## ไฟล์ที่แก้ไข

### Frontend
1. `frontend/src/components/CreateTrainJobModal.tsx` - เพิ่ม scale selector
2. `frontend/src/components/JobConfiguration.tsx` - แสดง partitions + scale
3. `frontend/src/pages/JobDetailPage.tsx` - ส่ง props ไปยัง JobConfiguration
4. `frontend/src/types/index.ts` - เพิ่ม model_scale ใน interfaces

### Backend
1. `backend/app/services/weight_storage.py` - เพิ่ม model_scale parameter
2. `backend/app/services/ultra_trainer.py` - ส่ง model_scale ไปยัง weight storage

## การใช้งาน

### สร้าง Training Job
1. เลือก Model
2. **เลือก Scale** (n/s/m/l/x) ⭐ ใหม่
3. เลือก Dataset
4. เลือก Partitions
5. ตั้งค่า Config
6. Start Training

### ดู Job Details
- Configuration section จะแสดง:
  - Model Scale (highlighted)
  - Partitions ที่ใช้พร้อม splits
  - Dataset name (ไม่ใช่ URL path)

### Weight Management
- Weights จะมี `model_scale` field
- สามารถ filter by scale ได้ (future feature)
- Lineage tracking รวม scale ด้วย

## Next Steps (Optional)

1. **Weight List Page**: แสดง scale badge ใน weight list
2. **Weight Filter**: Filter weights by scale
3. **Auto Scale Detection**: อ่าน scale จาก model YAML
4. **Scale Comparison**: เปรียบเทียบ performance ระหว่าง scales
