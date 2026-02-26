# YAML Import Feature

## Overview
ฟีเจอร์ import YAML ช่วยให้ผู้ใช้สามารถนำเข้าไฟล์ YAML จาก Ultralytics (YOLOv8, YOLO11, etc.) และแปลงเป็น graph format ใน Model Designer ได้โดยอัตโนมัติ

## Features

### Supported Model Types
- ✅ **Detection models** - YOLOv8, YOLO11 with multi-input heads (Concat, Detect)
- ✅ **Classification models** - Sequential head with Classify
- ✅ **Segmentation models** - Segment head
- ✅ **Pose models** - Pose head
- ✅ **OBB models** - Oriented bounding box detection

### Supported Layer Types
- **Backbone layers**: Conv, C2f, C3, SPPF, SPP, etc.
- **Head layers**: Detect, Classify, Segment, Pose, OBB
- **PyTorch nn blocks**: nn.Upsample, nn.Conv2d, nn.Linear, etc.
- **Multi-input layers**: Concat, Detect (with skip connections)
- **Sequential layers**: Classify (single chain)

## Backend Implementation

### File: `backend/app/services/yaml_to_graph.py`

Main functions:
- `yaml_to_graph(yaml_content)` - Convert YAML to graph format
- `_normalize_layers(layers)` - Convert YAML arrays to dict format
- `_resolve_from(from_val, current_idx)` - Resolve layer connections
- `_create_node(node_id, layer, layer_idx, section)` - Create graph nodes

### API Endpoints

#### POST `/api/models/import/yaml`
Import YAML content and create new model.

**Request:**
```json
{
  "yaml_content": "nc: 80\nbackbone:\n  - [-1, 1, Conv, [64, 3, 2]]\n...",
  "name": "Imported Model",
  "task": "detect"
}
```

**Response:**
```json
{
  "model_id": "abc123",
  "name": "Imported Model",
  "task": "detect",
  "message": "YAML imported successfully",
  "graph": {
    "nodes": [...],
    "edges": [...]
  }
}
```

#### POST `/api/models/import/yaml/file`
Import YAML file upload.

**Parameters:**
- `file`: YAML file (multipart/form-data)
- `name`: Model name (optional)
- `task`: Task type (optional, default: "detect")

## Frontend Implementation

### Component: `frontend/src/components/ImportYAMLModal.tsx`

Features:
- 📁 **File upload** - Drag & drop or click to upload .yaml/.yml files
- 📝 **Paste content** - Direct YAML content input
- ⚙️ **Model configuration** - Set name and task type
- ✅ **Validation** - Real-time error checking
- 🎨 **Modern UI** - Beautiful modal with Tailwind CSS

### Usage in Model Designer

1. Click **Import** button in toolbar
2. Choose method:
   - Upload YAML file, or
   - Paste YAML content
3. Set model name and task type
4. Click **Import YAML**
5. Model loads automatically in graph editor

## Graph Conversion Logic

### Node Creation
Each layer becomes a graph node with:
- **ID**: Unique identifier
- **Module**: Layer type (Conv, C2f, Detect, etc.)
- **Args**: Layer arguments
- **Repeats**: Number of repetitions
- **Position**: Auto-calculated (x, y) coordinates
- **Section**: "backbone" or "head"

### Edge Creation
Connections are created based on `from` field:
- `-1`: Previous layer (sequential)
- `[index]`: Specific layer by index
- `[-1, 6]`: Multiple inputs (e.g., Concat)
- `[15, 18, 21]`: Multi-scale inputs (e.g., Detect)

### Example: YOLOv8 Classification

**Input YAML:**
```yaml
nc: 1000
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
head:
  - [-1, 1, Classify, [nc]]
```

**Output Graph:**
- 4 nodes (3 backbone + 1 head)
- 3 edges (sequential connections)
- Sequential head structure

### Example: YOLOv8 Detection

**Input YAML:**
```yaml
nc: 80
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  # ... more layers
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  # ... more layers
  - [[15, 18, 21], 1, Detect, [nc]]
```

**Output Graph:**
- 23 nodes (10 backbone + 13 head)
- 28 edges (sequential + skip connections)
- Multi-input Detect head with 3 scale inputs

## Testing

### Test Files Created
- `test_yolov8_cls.yaml` - Classification model
- `test_yolov8_detect.yaml` - Detection model

### Test Results
```
✓ YOLOv8 Classification: 10 nodes, 9 edges
✓ YOLOv8 Detection: 23 nodes, 28 edges
✓ Sequential head: Classify from=-1
✓ Multi-input head: Detect from=[15, 18, 21]
✓ Multi-input layers: 5 (Concat + Detect)
```

## Benefits

1. **Fast prototyping** - Import existing YAML configs instantly
2. **Visual editing** - Convert YAML to interactive graph
3. **Learning tool** - Understand model architecture visually
4. **Compatibility** - Works with all Ultralytics YAML formats
5. **Flexibility** - Edit imported models in graph editor

## Usage Examples

### Import YOLOv8n Detection
```python
# Via API
import requests

yaml_content = open('yolov8n.yaml').read()
response = requests.post('/api/models/import/yaml', json={
    'yaml_content': yaml_content,
    'name': 'YOLOv8n',
    'task': 'detect'
})

model_id = response.json()['model_id']
```

### Import via Frontend
1. Open Model Designer
2. Click **Import** button
3. Upload `yolov8n.yaml`
4. Set name: "YOLOv8n"
5. Set task: "Detection"
6. Click **Import YAML**
7. Edit in graph editor

## Technical Details

### Node Positioning
- **Backbone**: Column 0-2, rows 0-N
- **Head**: Column 3-5, rows offset from backbone
- **Spacing**: 300px horizontal, 120px vertical
- **Auto-layout**: Prevents overlapping nodes

### Edge Resolution
```python
def _resolve_from(from_val, current_idx):
    if from_val == -1:
        return [current_idx - 1]  # Previous layer
    elif isinstance(from_val, list):
        return [resolve_index(f, current_idx) for f in from_val]
    else:
        return [from_val]  # Absolute index
```

### Multi-input Handling
- **Concat**: `[[-1, 6], 1, Concat, [1]]` → edges from layer -1 and layer 6
- **Detect**: `[[15, 18, 21], 1, Detect, [nc]]` → edges from layers 15, 18, 21

## Future Enhancements

- [ ] Export graph back to YAML
- [ ] Validate imported YAML before conversion
- [ ] Support for custom modules in YAML
- [ ] Batch import multiple YAML files
- [ ] Import from URL
- [ ] Preview graph before import

## Files Modified

### Backend
- `backend/app/services/yaml_to_graph.py` - New parser
- `backend/app/controllers/model_controller.py` - Import endpoints
- `backend/app/services/api.ts` - Import API method

### Frontend
- `frontend/src/components/ImportYAMLModal.tsx` - New modal component
- `frontend/src/pages/ModelDesignerPage.tsx` - Import button & handler
- `frontend/src/services/api.ts` - Import API client

## Conclusion

ฟีเจอร์ YAML import ทำให้ Model Designer สามารถรองรับการนำเข้าโมเดลจาก Ultralytics ได้อย่างสมบูรณ์ รองรับทั้ง sequential head (Classification) และ multi-input head (Detection, Segmentation) พร้อมแปลงเป็น graph format ที่สามารถแก้ไขได้ใน visual editor
