import torch
import torch.nn.functional as F

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    boxes1: (N, 4) in format (x1, y1, x2, y2)
    boxes2: (M, 4) in format (x1, y1, x2, y2)
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-Maximum Suppression.
    boxes: (N, 4) x1,y1,x2,y2
    scores: (N,)
    Returns: keep_indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
            
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        ids = (ovr <= iou_threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
            
        # ids is indices into order[1:], so we need to map back
        order = order[ids + 1] if ids.ndim > 0 else order[ids + 1].unsqueeze(0)
        
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def encode_target(targets: list[dict], num_classes: int, grid_size: int, device: torch.device) -> torch.Tensor:
    """
    Encode list of targets into a grid tensor for loss calculation.
    targets: List of dicts, each has 'boxes' (N, 4) in (x,y,w,h, norm 0-1) and 'labels' (N,)
    Output: (B, 5+C, S, S)
    """
    B = len(targets)
    S = grid_size
    C = num_classes
    
    # Shape: (B, 5+C, S, S)
    # 5: x, y, w, h, obj_conf
    target_tensor = torch.zeros((B, 5 + C, S, S), device=device)
    
    for b in range(B):
        t_boxes = targets[b]['boxes'] # (N, 4) xywh norm
        t_labels = targets[b]['labels'] # (N,)
        
        for i in range(len(t_boxes)):
            box = t_boxes[i]
            cls = int(t_labels[i])
            
            # Box is x,y,w,h normalized
            gx = box[0] * S
            gy = box[1] * S
            gw = box[2] * S
            gh = box[3] * S
            
            # Grid cell indices
            gj = int(gx)
            gi = int(gy)
            
            if 0 <= gj < S and 0 <= gi < S:
                # x,y relative to cell
                tx = gx - gj
                ty = gy - gi
                
                # w,h relative to image (or log encoded)
                # Let's use direct w,h relative to image (box[2], box[3]) for simplicity in loss
                # But here we store what the model PREDICTS.
                # Model predicts raw values.
                # Let's say model output is Sigmoid(tx), Sigmoid(ty), Sigmoid(tw), Sigmoid(th) -> 0-1
                # If we want w,h to be 0-1 relative to image, then target is box[2], box[3].
                
                # Assign to tensor
                # 0: x (cell relative)
                # 1: y (cell relative)
                # 2: w (image relative)
                # 3: h (image relative)
                # 4: obj (1.0)
                # 5..: class (one hot)
                
                target_tensor[b, 0, gi, gj] = tx
                target_tensor[b, 1, gi, gj] = ty
                target_tensor[b, 2, gi, gj] = box[2]
                target_tensor[b, 3, gi, gj] = box[3]
                target_tensor[b, 4, gi, gj] = 1.0
                if 0 <= cls < C:
                    target_tensor[b, 5 + cls, gi, gj] = 1.0
                    
    return target_tensor

def decode_prediction(output: torch.Tensor, conf_threshold: float = 0.1) -> list[torch.Tensor]:
    """
    Convert (B, 5+C, S, S) -> List of (N, 6) [x1, y1, x2, y2, score, label]
    """
    B, _, S, _ = output.shape
    num_classes = output.shape[1] - 5
    
    output = output.permute(0, 2, 3, 1) # B, S, S, 5+C
    
    pred_boxes = []
    
    for b in range(B):
        # Sigmoid for x, y, w, h, obj, probs
        # Assuming output is logits? Or already activated?
        # Usually model output is logits. We allow DetectionLoss to handle activation or model.
        # Let's assume model output is logits, so apply sigmoid here
        # But wait, if model ends with Conv2d, it's linear.
        
        out = output[b].sigmoid() 
        
        # Filter by obj confidence
        mask = out[..., 4] > conf_threshold
        if not mask.any():
            pred_boxes.append(torch.empty((0, 6), device=output.device))
            continue
            
        idxs = mask.nonzero() # (N, 2) -> grid_y, grid_x
        
        # Get values
        vals = out[idxs[:,0], idxs[:,1]] # (N, 5+C)
        
        tx = vals[:, 0]
        ty = vals[:, 1]
        tw = vals[:, 2]
        th = vals[:, 3]
        conf = vals[:, 4]
        cls_probs = vals[:, 5:]
        
        # Convert to global coords
        # x = (tx + grid_x) / S
        # y = (ty + grid_y) / S
        grid_x = idxs[:, 1].float()
        grid_y = idxs[:, 0].float()
        
        cx = (tx + grid_x) / S
        cy = (ty + grid_y) / S
        w = tw # w is 0-1 relative to image
        h = th
        
        # Convert cx,cy,w,h -> x1,y1,x2,y2
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        # Class label
        cls_scores, cls_ids = cls_probs.max(dim=1)
        final_scores = conf * cls_scores
        
        # (N, 6)
        preds = torch.stack([x1, y1, x2, y2, final_scores, cls_ids.float()], dim=1)
        pred_boxes.append(preds)
        
    return pred_boxes

class DetectionLoss(torch.nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, preds, targets):
        """
        preds: (B, 5+C, S, S)
        targets: (B, 5+C, S, S)
        """
        # 0:x, 1:y, 2:w, 3:h, 4:conf, 5+:class
        
        # Apply activations to preds to match target domain
        # x, y, conf, class -> Sigmoid
        # w, h -> Allow raw? Or sigmoid?
        # In encode_target, we decided:
        # target x,y are 0-1 relative to cell.
        # target w,h are 0-1 relative to image.
        
        # We assume model output is logits.
        p_xy = preds[:, 0:2, :, :].sigmoid()
        p_wh = preds[:, 2:4, :, :].sigmoid() # constrain w,h to 0-1 image size
        p_conf = preds[:, 4:5, :, :].sigmoid()
        p_cls = preds[:, 5:, :, :].sigmoid() # Multi-label or Softmax? YOLO uses sigmoid usually.
        
        # Targets
        t_xy = targets[:, 0:2, :, :]
        t_wh = targets[:, 2:4, :, :]
        t_conf = targets[:, 4:5, :, :]
        t_cls = targets[:, 5:, :, :]
        
        # Masks
        obj_mask = t_conf > 0.5
        noobj_mask = ~obj_mask
        
        # 1. Coordinate Loss (only for objects)
        loss_xy = F.mse_loss(p_xy[obj_mask.expand_as(p_xy)], t_xy[obj_mask.expand_as(t_xy)], reduction='sum')
        loss_wh = F.mse_loss(p_wh[obj_mask.expand_as(p_wh)], t_wh[obj_mask.expand_as(t_wh)], reduction='sum')
        
        # 2. Objectness Loss
        loss_conf_obj = F.mse_loss(p_conf[obj_mask], t_conf[obj_mask], reduction='sum')
        loss_conf_noobj = F.mse_loss(p_conf[noobj_mask], t_conf[noobj_mask], reduction='sum')
        loss_conf = loss_conf_obj + self.lambda_noobj * loss_conf_noobj
        
        # 3. Class Loss
        loss_cls = F.mse_loss(p_cls[obj_mask.expand_as(p_cls)], t_cls[obj_mask.expand_as(t_cls)], reduction='sum')
        
        # Total
        batch_size = preds.size(0)
        total_loss = (self.lambda_coord * (loss_xy + loss_wh) + loss_conf + loss_cls) / batch_size
        return total_loss
