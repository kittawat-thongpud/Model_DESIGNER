import base64
import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ..storage import load_model, WEIGHTS_DIR
from .codegen import generate_code
from . import detection_utils
from ..schemas.model_schema import PredictResponse, DetectionBox

def run_prediction(model_id: str, image_base64: str, weight_id: str = None) -> PredictResponse:
    # 1. Load model graph and generate code
    meta, graph = load_model(model_id)
    class_name, code = generate_code(graph)
    
    # Identify task type from Input node
    task_type = "classification"
    input_node = next((n for n in graph.nodes if n.type == "Input"), None)
    if input_node and input_node.params.get("task_type") == "detection":
        task_type = "detection"
    # Fallback to dataset check if task_type not explicitly set
    elif input_node and input_node.params.get("datasetSource") == "coco":
        task_type = "detection"

    # 2. Instantiate model
    namespace = {}
    exec(code, namespace)
    ModelClass = namespace[class_name]
    model = ModelClass()
    
    # 3. Load weights
    # If weight_id not provided, try to find the latest weight for this model
    # For now, let's assume if model_id.pt exists in WEIGHTS_DIR, we use it
    # Or model_id/latest.pt?
    # The current storage uses weight_id.pt
    if not weight_id:
        # Search for any weight associated with this model?
        # A simple fallback: check if WEIGHTS_DIR/model_id.pt exists
        weight_path = WEIGHTS_DIR / f"{model_id}.pt"
    else:
        weight_path = WEIGHTS_DIR / f"{weight_id}.pt"
        
    if weight_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(str(weight_path), map_location=device))
        model.to(device)
    else:
        # If no weights, we still run with random init (for testing UI)
        device = torch.device("cpu")
        model.to(device)
    
    model.eval()

    # 4. Pre-process image
    # Remove data:image/png;base64, prefix if present
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    # Basic Resize based on Input node params
    h = input_node.params.get("Height", 224) if input_node else 224
    w = input_node.params.get("Width", 224) if input_node else 224
    c = input_node.params.get("Channels", 3) if input_node else 3
    
    # Simple transform matches training skeleton
    tfm = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if c == 1 else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = tfm(img).unsqueeze(0).to(device)

    # 5. Inference
    with torch.no_grad():
        output = model(input_tensor)

    # 6. Post-process
    response = PredictResponse(model_id=model_id, task_type=task_type)
    
    if task_type == "classification":
        probs = torch.softmax(output, dim=1)
        conf, class_id = torch.max(probs, dim=1)
        response.class_id = int(class_id.item())
        response.confidence = float(conf.item())
        
        # Try to find class name (e.g. from dataset info)
        # For now, let's just return IDs or look up common names
        dataset_name = input_node.params.get("datasetSource") if input_node else "mnist"
        from .trainer import _CLASS_NAMES
        names = _CLASS_NAMES.get(dataset_name, [])
        if response.class_id < len(names):
            response.class_name = names[response.class_id]
            
    elif task_type == "detection":
        # boxes is list of (N, 6) [x1, y1, x2, y2, score, label]
        raw_boxes = detection_utils.decode_prediction(output, conf_threshold=0.1)
        if len(raw_boxes) > 0:
            batch_boxes = raw_boxes[0] # B=1
            
            # Run NMS
            keep = detection_utils.nms(batch_boxes[:, :4], batch_boxes[:, 4], iou_threshold=0.45)
            final_boxes = batch_boxes[keep]
            
            # Map back to image scale (normalized 0-1 currently)
            # Actually decode_prediction returns normalized 0-1
            
            response.boxes = []
            dataset_name = input_node.params.get("datasetSource") if input_node else "coco"
            from .trainer import _CLASS_NAMES
            names = _CLASS_NAMES.get(dataset_name, [])
            
            for b in final_boxes:
                label_id = int(b[5].item())
                label_name = names[label_id] if label_id < len(names) else str(label_id)
                
                response.boxes.append(DetectionBox(
                    x1=float(b[0].item()),
                    y1=float(b[1].item()),
                    x2=float(b[2].item()),
                    y2=float(b[3].item()),
                    score=float(b[4].item()),
                    label_id=label_id,
                    label_name=label_name
                ))
    
    return response
