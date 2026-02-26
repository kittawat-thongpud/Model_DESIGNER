# Fix Summary: Ultralytics Strict Config Validation

## Root Cause (ที่แท้จริง)

Ultralytics ตั้งแต่ v8+ ใช้ **strict config validation** - ตรวจสอบทุก key ที่ส่งเข้าไปว่าอยู่ใน `default.yaml` schema หรือไม่

```bash
# ตรวจสอบ config ที่รองรับได้ด้วย
yolo cfg
```

Custom parameters เช่น `sample_per_class`, `record_gradients` **ไม่อยู่ใน schema** → ถ้าส่งเข้าไปจะ throw error ทันที

## ข้อผิดพลาดในการแก้ไขครั้งก่อน

เราพยายามแก้โดย:
1. ✅ กรอง custom params ออกจาก config ก่อนส่งให้ `super().__init__()`
2. ❌ แต่แล้ว **inject กลับเข้า `self.args`** หลัง init

```python
# ❌ วิธีที่ผิด
super().__init__(cfg, clean_overrides, _callbacks)
self.args.sample_per_class = self.sample_per_class  # ← Ultralytics จะ validate args นี้!
```

ปัญหา: Ultralytics validate `self.args` ในหลายจุด รวมถึงตอนเรียก `model.train()` ทำให้เจอ error อยู่ดี

## วิธีแก้ที่ถูกต้อง

**ไม่ต้อง inject custom params เข้า `self.args` เลย** - เก็บเป็น instance variables แยกต่างหาก

### 1. เก็บ custom params เป็น instance variables
```python
class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # Extract custom params
        self.sample_per_class = clean_overrides.pop('sample_per_class', 0)
        
        # Filter invalid keys
        for key in INVALID_KEYS:
            cfg.pop(key, None)
        
        # Pass clean config to parent
        super().__init__(cfg, clean_overrides, _callbacks)
        
        # ✅ DO NOT inject into self.args
        # Keep as instance variable only: self.sample_per_class
```

### 2. ส่งค่าผ่าน method parameters แทน
```python
class CustomValidator(DetectionValidator):
    def __init__(self, ..., sample_per_class=0):  # ← รับเป็น parameter
        super().__init__(...)
        self.sample_per_class = sample_per_class  # ← ไม่อ่านจาก args

class CustomDetectionTrainer(DetectionTrainer):
    def get_validator(self):
        return CustomValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),  # ← args ไม่มี custom params
            sample_per_class=self.sample_per_class  # ← ส่งแยกต่างหาก
        )
```

## การเปลี่ยนแปลง

### ไฟล์: `backend/app/services/custom_trainer.py`

**1. CustomValidator - รับ parameter แยก**
```python
# Before
def __init__(self, ..., args=None, ...):
    super().__init__(...)
    self.sample_per_class = getattr(args, 'sample_per_class', 0)  # ❌

# After
def __init__(self, ..., args=None, ..., sample_per_class=0):  # ✅
    super().__init__(...)
    self.sample_per_class = sample_per_class  # ✅
```

**2. CustomDetectionTrainer - ไม่ inject เข้า args**
```python
# Before
super().__init__(cfg, clean_overrides, _callbacks)
self.args.sample_per_class = self.sample_per_class  # ❌

# After
super().__init__(cfg, clean_overrides, _callbacks)
# DO NOT inject into self.args  # ✅
# Keep as instance variable only
```

**3. get_validator - ส่งค่าแยก**
```python
# Before
return CustomValidator(
    self.test_loader,
    args=copy(self.args),  # args มี sample_per_class ❌
)

# After
return CustomValidator(
    self.test_loader,
    args=copy(self.args),  # args สะอาด ✅
    sample_per_class=self.sample_per_class  # ส่งแยก ✅
)
```

## ทำไมวิธีนี้ได้ผล

1. **Custom params ไม่เข้า Ultralytics validation**
   - ไม่อยู่ใน `cfg` → ไม่ validate ตอน `__init__()`
   - ไม่อยู่ใน `self.args` → ไม่ validate ตอน `model.train()`

2. **Functionality ยังคงทำงานได้**
   - เก็บเป็น instance variables: `self.sample_per_class`
   - ส่งผ่าน method parameters: `get_validator()`
   - Validator ได้รับค่าและใช้งานได้ปกติ

3. **ไม่ต้อง monkey-patch Ultralytics**
   - ไม่ต้อง patch `cfg.entrypoint`
   - ไม่ต้องกังวลเรื่อง CLI mode
   - แค่ไม่ส่ง invalid keys เข้าไป

## สรุป

### ❌ สิ่งที่ไม่ควรทำ
- ส่ง custom params เข้า `model.train(**kwargs)`
- Inject custom params เข้า `self.args` หลัง init
- อ่าน custom params จาก `args` object

### ✅ สิ่งที่ควรทำ
- เก็บ custom params เป็น instance variables
- กรอก custom params ออกจาก config ก่อนส่งให้ Ultralytics
- ส่งค่าผ่าน method parameters แทน

## ไฟล์ที่แก้ไข

1. `backend/app/services/custom_trainer.py`
   - ลบ `self.args.sample_per_class = ...`
   - เพิ่ม `sample_per_class` parameter ใน `CustomValidator.__init__()`
   - ส่ง `sample_per_class` ผ่าน `get_validator()`

2. `backend/app/main.py` (จากการแก้ไขก่อนหน้า)
   - `sys.argv = []`

3. `backend/app/services/ultra_trainer.py` (จากการแก้ไขก่อนหน้า)
   - `sys.argv = []` + entrypoint patching

## Verification

```bash
# ควรทำงานได้โดยไม่มี error
python3 -m py_compile backend/app/services/custom_trainer.py
```

Training ควรทำงานได้โดยไม่เจอ:
- ✅ `'sample_per_class' is not a valid YOLO argument`
- ✅ `'session' is not a valid YOLO argument`
- ✅ `Arguments received: ['yolo', '']`
