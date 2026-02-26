"""
OneToFewDetect - One-to-Few Matching Detection Head for HSG-DET

This module implements the detection head described in the HSG-DET presentation:
- One-to-few matching (2-3 predictions per object)
- Dynamic assignment using Hungarian algorithm
- Optional NMS (fallback only)
- Structured predictions

Architecture:
    Input: Multi-scale features [P3, P4, P5]
    Output: K queries with (box, class, objectness)
    Matching: Hungarian matching during training
    Inference: Top-K selection + optional NMS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ultralytics.nn.modules import Conv, DFL
from ultralytics.utils.tal import dist2bbox, make_anchors


class OneToFewDetect(nn.Module):
    """
    One-to-Few Detection Head for HSG-DET
    
    Args:
        nc (int): Number of classes
        num_queries (int): Number of object queries (default: 300)
        k_per_object (int): Max predictions per object (default: 3)
        ch (tuple): Input channels from [P3, P4, P5]
    """
    
    dynamic = False  # Force grid-based anchors for compatibility
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    
    def __init__(self, nc=80, num_queries=300, k_per_object=3, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers (3: P3, P4, P5)
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.num_queries = num_queries
        self.k_per_object = k_per_object
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        # Query embeddings (learnable)
        self.query_embed = nn.Embedding(num_queries, 256)
        
        # Shared convs for multi-scale features
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )
        
        # Query decoder (cross-attention)
        self.decoder = QueryDecoder(
            d_model=256,
            num_queries=num_queries,
            num_layers=3,
            nhead=8
        )
        
        # Prediction heads
        self.box_head = nn.Linear(256, 4)
        self.class_head = nn.Linear(256, nc)
        self.objectness_head = nn.Linear(256, 1)
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: List of feature maps [P3, P4, P5]
            
        Returns:
            During training: (predictions, query_features)
            During inference: predictions (after NMS if needed)
        """
        # Extract multi-scale features
        shape = x[0].shape
        
        # Process each scale with standard YOLO heads (for feature extraction)
        box_feats = []
        cls_feats = []
        for i in range(self.nl):
            box_feats.append(self.cv2[i](x[i]))
            cls_feats.append(self.cv3[i](x[i]))
        
        # Flatten and concatenate multi-scale features
        # This creates a feature map that queries will attend to
        B = shape[0]
        feature_maps = []
        for i in range(self.nl):
            h, w = x[i].shape[2:]
            feat = x[i].flatten(2).transpose(1, 2)  # [B, H*W, C]
            feature_maps.append(feat)
        
        # Concatenate all scales
        all_features = torch.cat(feature_maps, dim=1)  # [B, N_total, C]
        
        # Get query embeddings
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, 256]
        
        # Decode queries with cross-attention to features
        query_features = self.decoder(queries, all_features)  # [B, num_queries, 256]
        
        # Predict from queries
        pred_boxes = self.box_head(query_features)  # [B, num_queries, 4]
        pred_logits = self.class_head(query_features)  # [B, num_queries, nc]
        pred_objectness = self.objectness_head(query_features)  # [B, num_queries, 1]
        
        if self.training:
            # Return predictions for loss computation
            return {
                'pred_boxes': pred_boxes,
                'pred_logits': pred_logits,
                'pred_objectness': pred_objectness,
                'query_features': query_features
            }
        else:
            # Inference: select top-K confident predictions
            objectness = pred_objectness.sigmoid().squeeze(-1)  # [B, num_queries]
            class_probs = pred_logits.sigmoid()  # [B, num_queries, nc]
            
            # Get max class probability for each query
            max_probs, pred_classes = class_probs.max(dim=-1)  # [B, num_queries]
            
            # Combined confidence score
            confidence = objectness * max_probs  # [B, num_queries]
            
            # Select top predictions
            top_k = min(self.num_queries, 100)  # Limit to top 100
            top_conf, top_idx = confidence.topk(top_k, dim=-1)  # [B, top_k]
            
            # Gather top predictions
            batch_idx = torch.arange(B, device=x[0].device).unsqueeze(1).expand(-1, top_k)
            top_boxes = pred_boxes[batch_idx, top_idx]  # [B, top_k, 4]
            top_classes = pred_classes[batch_idx, top_idx]  # [B, top_k]
            top_probs = max_probs[batch_idx, top_idx]  # [B, top_k]
            
            # Convert to YOLO format: [x, y, w, h] -> [x1, y1, x2, y2]
            # Assuming boxes are in normalized [0, 1] format
            boxes_xyxy = self._box_cxcywh_to_xyxy(top_boxes)
            
            # Stack predictions: [B, top_k, 4+1+nc]
            predictions = torch.cat([
                boxes_xyxy,
                top_conf.unsqueeze(-1),
                F.one_hot(top_classes, self.nc).float() * top_probs.unsqueeze(-1)
            ], dim=-1)
            
            return predictions
    
    def _box_cxcywh_to_xyxy(self, boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes (compatibility method)"""
        return self._box_cxcywh_to_xyxy(bboxes)


class QueryDecoder(nn.Module):
    """
    Query Decoder with Cross-Attention
    
    Queries attend to multi-scale feature maps to extract object information
    """
    
    def __init__(self, d_model=256, num_queries=300, num_layers=3, nhead=8):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, features):
        """
        Args:
            queries: [B, num_queries, d_model]
            features: [B, N_features, d_model]
            
        Returns:
            decoded_queries: [B, num_queries, d_model]
        """
        output = queries
        for layer in self.layers:
            output = layer(output, features)
        return self.norm(output)


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention"""
    
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, queries, features):
        # Self-attention among queries
        q = self.norm1(queries)
        queries = queries + self.dropout(self.self_attn(q, q, q)[0])
        
        # Cross-attention to features
        q = self.norm2(queries)
        queries = queries + self.dropout(self.cross_attn(q, features, features)[0])
        
        # FFN
        q = self.norm3(queries)
        queries = queries + self.dropout(self.ffn(q))
        
        return queries


class HungarianMatcher:
    """
    Hungarian Matcher for one-to-few assignment
    
    Assigns ground truth objects to predicted queries with minimal cost
    Allows k_per_object predictions per ground truth
    """
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, k_per_object=3):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.k_per_object = k_per_object
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_boxes', 'pred_logits'
            targets: list of dicts with 'boxes', 'labels'
            
        Returns:
            List of (pred_idx, target_idx) tuples for each image
        """
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions
        out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()  # [B*Q, nc]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [B*Q, 4]
        
        # Concatenate targets
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])
        
        # Compute classification cost
        cost_class = -out_prob[:, tgt_ids]  # [B*Q, num_targets]
        
        # Compute L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute GIoU cost
        cost_giou = -self._generalized_box_iou(
            self._box_cxcywh_to_xyxy(out_bbox),
            self._box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        # Split by batch and perform Hungarian matching
        sizes = [len(t['boxes']) for t in targets]
        indices = []
        start_idx = 0
        for i, (c, size) in enumerate(zip(C, sizes)):
            if size == 0:
                indices.append((torch.tensor([]), torch.tensor([])))
                continue
                
            c_i = c[:, start_idx:start_idx + size]
            
            # Allow k predictions per target
            pred_idx_list = []
            tgt_idx_list = []
            
            for k in range(self.k_per_object):
                # Find best assignment
                pred_idx, tgt_idx = linear_sum_assignment(c_i.numpy())
                pred_idx_list.append(torch.as_tensor(pred_idx, dtype=torch.int64))
                tgt_idx_list.append(torch.as_tensor(tgt_idx, dtype=torch.int64))
                
                # Mask out assigned predictions for next iteration
                c_i[pred_idx, :] = float('inf')
            
            # Concatenate all k assignments
            indices.append((
                torch.cat(pred_idx_list),
                torch.cat(tgt_idx_list)
            ))
            
            start_idx += size
            
        return indices
    
    def _box_cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
    
    def _generalized_box_iou(self, boxes1, boxes2):
        """Compute GIoU between two sets of boxes"""
        eps = 1e-7
        wh1 = (boxes1[:, 2:4] - boxes1[:, 0:2]).clamp(min=0)
        wh2 = (boxes2[:, 2:4] - boxes2[:, 0:2]).clamp(min=0)
        area1 = wh1[:, 0] * wh1[:, 1]
        area2 = wh2[:, 0] * wh2[:, 1]
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=eps)
        
        # Enclosing box
        lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh_enc = (rb_enc - lt_enc).clamp(min=0)
        area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]
        
        giou = iou - (area_enc - union) / area_enc.clamp(min=eps)
        return giou
