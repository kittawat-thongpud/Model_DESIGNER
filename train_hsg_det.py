# hsg_det.py
# HSG-DET single-file PyTorch implementation (trainable without ultralytics)
from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic building blocks
# -----------------------------
def conv_bn_act(in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True):
    if p is None:
        p = k // 2
    layers = [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class DWConv(nn.Module):
    """Depthwise separable conv to reduce params/cost."""
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = conv_bn_act(c1, c1, k=k, s=s, groups=c1)
        self.pw = conv_bn_act(c1, c2, k=1, s=1)

    def forward(self, x): return self.pw(self.dw(x))


class C2f(nn.Module):
    """Simplified cross-stage partial block with residuals (C2f-like)."""
    def __init__(self, c1, c2, n=1, use_dw=False):
        super().__init__()
        self.cv1 = conv_bn_act(c1, c2 // 2, k=1)
        self.cv2 = conv_bn_act(c1, c2 // 2, k=1)
        blocks = []
        for _ in range(n):
            blocks.append(DWConv(c2 // 2, c2 // 2) if use_dw else conv_bn_act(c2 // 2, c2 // 2, 3))
        self.m = nn.Sequential(*blocks)
        self.cv3 = conv_bn_act(c2, c2, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y2 = self.m(y2)
        return self.cv3(torch.cat((y1, y2), dim=1))


class SPPF(nn.Module):
    """SPPF: stack of maxpools (fast SPP)."""
    def __init__(self, c, k=5):
        super().__init__()
        c_ = c // 2
        self.cv1 = conv_bn_act(c, c_, 1)
        self.cv2 = conv_bn_act(c_ * (k // 2 + 1), c, 1)
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = x1
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([y1, y2, y3], dim=1))


# -----------------------------
# Sparse Global Block (SGB)
# -----------------------------
class SparseGlobalBlock(nn.Module):
    """
    Hybrid sparse self-attention block.
    Selects top-k tokens by L2 activation energy, attends among them, scatters back.
    """
    def __init__(self, c: int, k: int = 256):
        super().__init__()
        self.k = k
        self.c = c
        self.q_proj = nn.Conv2d(c, c, 1, bias=False)
        self.k_proj = nn.Conv2d(c, c, 1, bias=False)
        self.v_proj = nn.Conv2d(c, c, 1, bias=False)
        self.out_proj = nn.Conv2d(c, c, 1, bias=False)
        self.norm = nn.LayerNorm(c)
        self._scale = c ** -0.5

    def _attention_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sparse-attention delta (no residual). Used by both forward() and SparseGlobalBlockGated."""
        B, C, H, W = x.shape
        N = H * W
        k_actual = min(self.k, N)
        q = self.q_proj(x).view(B, C, N)
        kk = self.k_proj(x).view(B, C, N)
        v = self.v_proj(x).view(B, C, N)

        importance = x.view(B, C, N).float().pow(2).sum(dim=1)  # [B, N]
        importance = torch.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        topk_idx = torch.topk(importance, k_actual, dim=1).indices  # [B, k]
        idx_exp = topk_idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, k]

        q_sel = torch.gather(q, 2, idx_exp).transpose(1, 2).float()  # [B, k, C]
        k_sel = torch.gather(kk, 2, idx_exp).transpose(1, 2).float()
        v_sel = torch.gather(v, 2, idx_exp).transpose(1, 2).float()

        # Layer norm in fp32
        q_sel = F.layer_norm(q_sel, (C,), self.norm.weight.float() if self.norm.weight is not None else None,
                             self.norm.bias.float() if self.norm.bias is not None else None, self.norm.eps)
        k_sel = F.layer_norm(k_sel, (C,), self.norm.weight.float() if self.norm.weight is not None else None,
                             self.norm.bias.float() if self.norm.bias is not None else None, self.norm.eps)

        attn = torch.bmm(q_sel, k_sel.transpose(1, 2)) * float(self._scale)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn.clamp(min=-80.0, max=80.0)
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attended = torch.bmm(attn, v_sel)  # [B, k, C]
        attended = attended.to(q.dtype)

        out = v.clone()
        out.scatter_(2, idx_exp, attended.transpose(1, 2))
        out = out.view(B, C, H, W)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._attention_delta(x)


class SparseGlobalBlockGated(nn.Module):
    """Gated variant to stabilize warm-starts."""
    def __init__(self, c, k=256):
        super().__init__()
        self.block = SparseGlobalBlock(c, k=k)
        self.gate = nn.Parameter(torch.zeros(1))  # starts as identity

    def forward(self, x):
        # Call _attention_delta (not block.forward) to avoid double-residual:
        # block.forward = x+delta, so x + gate*(x+delta) would corrupt features.
        return x + self.gate * self.block._attention_delta(x)


# -----------------------------
# Neck: PAN-like + insert SGB at P4,P5
# -----------------------------
class SimplePANNeck(nn.Module):
    """
    Simple top-down then bottom-up (FPN + PAN) neck.
    Accepts P3,P4,P5 inputs and returns processed features at same scales.
    We keep channels configured to target sizes (256,512,1024).
    """
    def __init__(self, channels=(256, 512, 1024),
                 sgb_k_p4: int = 256, sgb_k_p5: int = 256,
                 use_gated: bool = True):
        super().__init__()
        c3, c4, c5 = channels
        # lateral convs to unify channels
        self.lateral5 = conv_bn_act(c5, c4, k=1)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f4 = C2f(c4 * 2, c4, n=3, use_dw=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f3 = C2f(c3 + c4, c3, n=3, use_dw=True)

        # bottom-up
        self.down3 = conv_bn_act(c3, c3, k=3, s=2)
        self.c2f34 = C2f(c3 + c4, c4, n=3, use_dw=True)
        self.down4 = conv_bn_act(c4, c4, k=3, s=2)
        self.c2f45 = C2f(c4 + c5, c5, n=3, use_dw=True)

        # SGB at P4 and P5
        if use_gated:
            self.sgb_p5 = SparseGlobalBlockGated(c5, k=sgb_k_p5)
            self.sgb_p4 = SparseGlobalBlockGated(c4, k=sgb_k_p4)
        else:
            self.sgb_p5 = SparseGlobalBlock(c5, k=sgb_k_p5)
            self.sgb_p4 = SparseGlobalBlock(c4, k=sgb_k_p4)

    def forward(self, p3, p4, p5):
        # p5 -> top-down
        p5_l = self.lateral5(p5)
        p5_up = F.interpolate(p5_l, size=p4.shape[-2:], mode='nearest')
        p4_td = torch.cat([p4, p5_up], dim=1)
        p4_td = self.c2f4(p4_td)

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_td = torch.cat([p3, p4_up], dim=1)
        p3_out = self.c2f3(p3_td)

        # bottom-up
        p3_down = self.down3(p3_out)
        p4_bu = torch.cat([p3_down, p4_td], dim=1)
        p4_out = self.c2f34(p4_bu)
        p4_down = self.down4(p4_out)
        p5_bu = torch.cat([p4_down, p5], dim=1)
        p5_out = self.c2f45(p5_bu)

        # Inject global context after local aggregation
        p5_out = p5_out + self.sgb_p5(p5_out)
        p4_out = p4_out + self.sgb_p4(p4_out)

        return p3_out, p4_out, p5_out


# -----------------------------
# Decoupled Head (Box / Cls / Obj)
# -----------------------------
class DecoupledHeadPerScale(nn.Module):
    """
    Per-scale decoupled head: separate conv stacks for box, cls, obj.
    """
    def __init__(self, in_ch, num_classes, hidden=256, num_layers=2):
        super().__init__()
        self.box_head = nn.Sequential(*[conv_bn_act(in_ch if i == 0 else hidden, hidden, k=3) for i in range(num_layers)],
                                      nn.Conv2d(hidden, 4, 1))  # xywh
        self.cls_head = nn.Sequential(*[conv_bn_act(in_ch if i == 0 else hidden, hidden, k=3) for i in range(num_layers)],
                                      nn.Conv2d(hidden, num_classes, 1))
        self.obj_head = nn.Sequential(*[conv_bn_act(in_ch if i == 0 else hidden, hidden, k=3) for i in range(num_layers)],
                                      nn.Conv2d(hidden, 1, 1))

    def forward(self, x):
        # x: [B, C, H, W]
        box = self.box_head(x)  # [B, 4, H, W]
        cls = self.cls_head(x)  # [B, num_classes, H, W]
        obj = self.obj_head(x)  # [B, 1, H, W]
        return box, cls, obj


class DecoupledHead(nn.Module):
    """Head combining three scales outputs and producing flattened predictions per-batch."""
    def __init__(self, channels=(256, 512, 1024), num_classes=80, head_hidden=256):
        super().__init__()
        self.num_classes = num_classes
        self.scales = len(channels)
        self.heads = nn.ModuleList([DecoupledHeadPerScale(c, num_classes, hidden=head_hidden) for c in channels])

    def forward(self, px: List[torch.Tensor]):
        # px: list of feature maps [p3, p4, p5]
        outs = []
        shapes = []
        for i, feat in enumerate(px):
            box, cls, obj = self.heads[i](feat)
            B, _, H, W = box.shape
            # flatten per cell: xywh, cls_logits, obj_logit
            box_f = box.view(B, 4, -1).permute(0, 2, 1)  # [B, N, 4]
            cls_f = cls.view(B, self.num_classes, -1).permute(0, 2, 1)  # [B, N, C]
            obj_f = obj.view(B, 1, -1).permute(0, 2, 1)  # [B, N, 1]
            pred = torch.cat([box_f, obj_f, cls_f], dim=2)  # [B, N, 4+1+C]
            outs.append(pred)
            shapes.append((H, W))
        # concatenate across scales -> candidate queue
        preds_all = torch.cat(outs, dim=1)  # [B, N_total, 5+C]
        return preds_all, shapes


# -----------------------------
# Utility: bbox ops
# -----------------------------
def xywh_to_xyxy(x):
    # x: [...,4] (cx,cy,w,h)
    xc, yc, w, h = x.unbind(-1)
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_iou(ci, cj, eps=1e-7):
    # inputs are xyxy
    x1 = torch.max(ci[..., 0], cj[..., 0])
    y1 = torch.max(ci[..., 1], cj[..., 1])
    x2 = torch.min(ci[..., 2], cj[..., 2])
    y2 = torch.min(ci[..., 3], cj[..., 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_i = (ci[..., 2] - ci[..., 0]).clamp(0) * (ci[..., 3] - ci[..., 1]).clamp(0)
    area_j = (cj[..., 2] - cj[..., 0]).clamp(0) * (cj[..., 3] - cj[..., 1]).clamp(0)
    union = area_i + area_j - inter + eps
    return inter / union


def ciou_loss(pred_xyxy, gt_xyxy, eps=1e-7):
    # pred_xyxy, gt_xyxy: [...,4]
    # returns 1 - CIoU
    # IoU
    iou = bbox_iou(pred_xyxy, gt_xyxy, eps)
    # centers
    pred_cx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
    pred_cy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
    gt_cx = (gt_xyxy[..., 0] + gt_xyxy[..., 2]) / 2
    gt_cy = (gt_xyxy[..., 1] + gt_xyxy[..., 3]) / 2
    center_dist2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
    # enclose box
    c_x1 = torch.min(pred_xyxy[..., 0], gt_xyxy[..., 0])
    c_y1 = torch.min(pred_xyxy[..., 1], gt_xyxy[..., 1])
    c_x2 = torch.max(pred_xyxy[..., 2], gt_xyxy[..., 2])
    c_y2 = torch.max(pred_xyxy[..., 3], gt_xyxy[..., 3])
    c_w = (c_x2 - c_x1).clamp(min=eps)
    c_h = (c_y2 - c_y1).clamp(min=eps)
    c2 = c_w ** 2 + c_h ** 2 + eps

    # aspect ratio term
    w_pred = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=eps)
    h_pred = (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=eps)
    w_gt = (gt_xyxy[..., 2] - gt_xyxy[..., 0]).clamp(min=eps)
    h_gt = (gt_xyxy[..., 3] - gt_xyxy[..., 1]).clamp(min=eps)
    v = (4 / (math.pi ** 2)) * ((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)) ** 2)
    with torch.no_grad():
        S = 1 - iou
        alpha = v / (S + v + eps)
    ciou = iou - (center_dist2 / c2) - alpha * v
    return 1 - ciou  # loss (lower better)


# -----------------------------
# Varifocal Loss (approx.)
# -----------------------------
class VarifocalLoss(nn.Module):
    """
    Approximate Varifocal Loss:
    - for positive sample target = iou (soft target)
    - for negative sample target = 0
    weight term uses focal-like scaling on predictions for negatives.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, target_score):
        # pred_logits: [N]
        # target_score: [N] in [0,1] (for pos: iou, for neg: 0)
        pred_sig = torch.sigmoid(pred_logits)
        weight = target_score * 1.0 + (1 - target_score) * (pred_sig ** self.gamma) * self.alpha
        loss = self.bce(pred_logits, target_score)
        return (weight * loss).sum() / max(1.0, target_score.numel())


# -----------------------------
# One-to-Few Dynamic Matcher
# -----------------------------
def K_schedule(epoch: int, total_epochs: int, K_max=10, K_min=1):
    """Linearly decay K from K_max -> K_min over training."""
    t = epoch / max(1, total_epochs - 1)
    K = int(round(K_max - (K_max - K_min) * t))
    return max(K_min, min(K_max, K))


def match_one2few(preds: torch.Tensor, gt_boxes: torch.Tensor, gt_classes: torch.Tensor,
                  epoch: int, total_epochs: int, alpha: float = 0.75, beta: float = 1.0,
                  K_max=10, K_min=1):
    """
    preds: [N_pred, 5+C] (xywh, obj, cls_logits...)
    gt_boxes: [N_gt, 4] in xyxy (image coords)
    gt_classes: [N_gt] int labels
    returns assignments: list of (pred_idx, gt_idx)
    Flow:
      - compute classification score (sigmoid) for each pred vs gt class
      - compute CIoU between predicted bbox and gt bbox
      - cost = alpha*(1-cls_score) + beta*(1-CIoU)
      - for each gt, pick top-K(t) preds with smallest cost
    """
    device = preds.device
    Np = preds.shape[0]
    Ng = gt_boxes.shape[0]
    if Ng == 0 or Np == 0:
        return torch.empty((0, 2), dtype=torch.long, device=device)

    # decode preds xywh->xyxy (assuming xywh absolute coordinates)
    pred_xywh = preds[:, :4]
    pred_xyxy = xywh_to_xyxy(pred_xywh)
    # pred class scores
    cls_logits = preds[:, 5:]  # [Np, C]
    obj_logits = preds[:, 4:5]
    cls_prob = torch.sigmoid(cls_logits)  # [Np, C]

    assignments = []
    Kt = K_schedule(epoch, total_epochs, K_max=K_max, K_min=K_min)
    # compute cost matrix Ng x Np
    with torch.no_grad():
        # expand to compute pairwise
        pred_exp = pred_xyxy.unsqueeze(0).expand(Ng, -1, -1)  # [Ng, Np, 4]
        gt_exp = gt_boxes.unsqueeze(1).expand(-1, Np, -1)  # [Ng, Np, 4]
        ciou = 1 - ciou_loss(pred_exp.reshape(-1, 4), gt_exp.reshape(-1, 4)).view(Ng, Np)  # CIoU similarity
        # class prob for each gt: pick predicted prob for that gt class
        gt_cls_expand = gt_classes.view(-1, 1).expand(-1, Np)  # [Ng, Np]
        cls_score = cls_prob[:, :].t().unsqueeze(0).expand(Ng, -1, -1)  # [Ng, Np, C] -> but heavy
        # Instead compute by indexing:
        cls_score_per = []
        for gi in range(Ng):
            c = int(gt_classes[gi].item())
            cls_score_per.append(cls_prob[:, c].unsqueeze(0))  # [1, Np]
        cls_score = torch.cat(cls_score_per, dim=0)  # [Ng, Np]

        cost = alpha * (1.0 - cls_score) + beta * (1.0 - ciou.clamp(0, 1))
        # For each gt, pick top-Kt smallest cost (i.e., best matches)
        topk_vals, topk_idx = torch.topk(-cost, k=min(Kt, Np), dim=1)  # using negative to get smallest cost
        # produce assignments
        for gi in range(Ng):
            preds_for_gt = topk_idx[gi]  # indices of preds
            gi_repeat = torch.full((preds_for_gt.shape[0],), gi, dtype=torch.long, device=device)
            assignments.append(torch.stack([preds_for_gt, gi_repeat], dim=1))
    if assignments:
        return torch.cat(assignments, dim=0)  # [M, 2] columns (pred_idx, gt_idx)
    return torch.empty((0, 2), dtype=torch.long, device=device)


# -----------------------------
# HSG-DET top-level model
# -----------------------------
class HSG_DET(nn.Module):
    """
    Top-level model:
    - backbone -> outputs P3,P4,P5 channels (256,512,1024)
    - neck: PAN + SGB at P4,P5
    - head: decoupled head (predicts per-cell: xywh,obj,cls_logits)
    - queue selection: we return all candidates (caller may top-k)
    """
    def __init__(self, num_classes: int = 80, channels=(256, 512, 1024),
                 sgb_k_p4=256, sgb_k_p5=256, head_hidden=256):
        super().__init__()
        self.num_classes = num_classes
        # --- backbone (simple CSP-like pyramid)
        # stem
        self.stem = nn.Sequential(conv_bn_act(3, 64, k=3, s=2),
                                  conv_bn_act(64, 128, k=3, s=2))
        # stages produce P3, P4, P5
        self.stage3 = nn.Sequential(C2f(128, channels[0], n=3, use_dw=True))
        self.down34 = conv_bn_act(channels[0], channels[1], k=3, s=2)
        self.stage4 = nn.Sequential(C2f(channels[1], channels[1], n=3, use_dw=True))
        self.down45 = conv_bn_act(channels[1], channels[2], k=3, s=2)
        self.stage5 = nn.Sequential(C2f(channels[2], channels[2], n=3, use_dw=True), SPPF(channels[2], k=5))

        # neck
        self.neck = SimplePANNeck(channels=channels, sgb_k_p4=sgb_k_p4, sgb_k_p5=sgb_k_p5)

        # head
        self.head = DecoupledHead(channels=channels, num_classes=num_classes, head_hidden=head_hidden)

        # losses
        self.vfl = VarifocalLoss(alpha=0.75, gamma=2.0)
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

        # loss weights
        self.lambda_box = 7.5
        self.lambda_cls = 0.5
        self.lambda_obj = 1.0

    def forward_backbone(self, x: torch.Tensor):
        # x: [B,3,H,W]
        s = self.stem(x)
        p3 = self.stage3(s)  # stride 8
        p4 = self.stage4(self.down34(p3))  # stride 16
        p5 = self.stage5(self.down45(p4))  # stride 32
        return p3, p4, p5

    def forward(self, imgs: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]] = None,
                epoch: int = 0, total_epochs: int = 300, topk_candidates: Optional[int] = None):
        """
        imgs: Bx3xH'xW' (absolute image coordinates expected for boxes)
        targets (optional): list length B, each dict with 'boxes' [Ng,4: xyxy], 'labels' [Ng]
        If targets provided -> compute losses with One-to-Few dynamic assignment
        Returns:
          if training: {'loss': total_loss, 'loss_components': {...}}
          else: predictions list per image (tensor Nx(5+C) in xywh coords normalized to image pixels)
        """
        device = imgs.device
        B = imgs.shape[0]
        p3, p4, p5 = self.forward_backbone(imgs)
        p3n, p4n, p5n = self.neck(p3, p4, p5)
        preds_all, shapes = self.head([p3n, p4n, p5n])  # [B, N_total, 5+C]
        # Predict format: [cx,cy,w,h, obj_logit, cls_logits...]

        # optional top-k candidate selection per image (global)
        if topk_candidates is not None:
            objs = torch.sigmoid(preds_all[:, :, 4])  # [B, N]
            k = min(topk_candidates, objs.shape[1])
            topk = torch.topk(objs, k, dim=1).indices  # [B, k]
            idx = topk.unsqueeze(-1).expand(-1, -1, preds_all.shape[2])
            preds_topk = torch.gather(preds_all, 1, idx)
        else:
            preds_topk = preds_all

        if self.training and targets is not None:
            # compute losses per image and sum/average
            total_box_loss = 0.0
            total_cls_loss = 0.0
            total_obj_loss = 0.0
            # flatten across batch for matching convenience
            batch_losses = {'box': 0.0, 'cls': 0.0, 'obj': 0.0}
            for bi in range(B):
                preds_b = preds_topk[bi]  # [N,5+C]
                t = targets[bi]
                gt_boxes = t.get('boxes', torch.zeros((0, 4), device=device))  # xyxy
                gt_labels = t.get('labels', torch.zeros((0,), dtype=torch.long, device=device))
                if gt_boxes.shape[0] == 0:
                    # all negatives -> objectness target 0
                    obj_logits = preds_b[:, 4]
                    obj_target = torch.zeros_like(obj_logits)
                    obj_loss = (self.bce_obj(obj_logits, obj_target)).mean()
                    total_obj_loss += obj_loss
                    continue

                # decode preds XYWH->XYXY (predates are absolute coords)
                pred_xywh = preds_b[:, :4]
                pred_xyxy = xywh_to_xyxy(pred_xywh)

                # match
                assignments = match_one2few(preds_b.detach(), gt_boxes.detach(), gt_labels.detach(),
                                           epoch=epoch, total_epochs=total_epochs, K_max=10, K_min=1)
                if assignments.numel() == 0:
                    # fallback: no matches
                    obj_logits = preds_b[:, 4]
                    obj_target = torch.zeros_like(obj_logits)
                    obj_loss = (self.bce_obj(obj_logits, obj_target)).mean()
                    total_obj_loss += obj_loss
                    continue

                # build targets per pred index
                Np = preds_b.shape[0]
                obj_target = torch.zeros((Np,), device=device)
                cls_target = torch.zeros((Np, self.num_classes), device=device)
                box_target = torch.zeros((Np, 4), device=device)
                pos_idx = assignments[:, 0]
                gt_idx = assignments[:, 1]
                # for positives: set obj_target=1, cls target = one-hot * iou (varifocal expects soft targets)
                for j in range(assignments.shape[0]):
                    pi = int(pos_idx[j].item())
                    gi = int(gt_idx[j].item())
                    # compute iou between this pred and that gt
                    pxyxy = pred_xyxy[pi:pi+1]
                    gxyxy = gt_boxes[gi:gi+1]
                    iou_val = bbox_iou(pxyxy, gxyxy).clamp(0, 1)[0]
                    obj_target[pi] = 1.0
                    lab = gt_labels[gi].long().item()
                    cls_target[pi, lab] = iou_val  # soft target = iou
                    box_target[pi] = gxyxy  # use gt xyxy as target for CIoU and l1 part

                # box loss: only on positives (convert pred_xyxy and box_target)
                pos_mask = obj_target > 0.5
                if pos_mask.sum() > 0:
                    pred_pos_xyxy = pred_xyxy[pos_mask]
                    gt_pos_xyxy = box_target[pos_mask]
                    l_box = ciou_loss(pred_pos_xyxy, gt_pos_xyxy).mean()
                else:
                    l_box = torch.tensor(0.0, device=device)

                # classification: varifocal on class logits (we use logits predicted)
                cls_logits = preds_b[:, 5:]
                # For varifocal, target per class is cls_target (soft) else 0
                # flatten
                l_cls = self.vfl(cls_logits.view(-1), cls_target.view(-1)) if cls_target.numel() > 0 else torch.tensor(0.0, device=device)

                # objectness: BCE against obj_target
                obj_logits = preds_b[:, 4]
                l_obj = (self.bce_obj(obj_logits, obj_target)).mean()

                total_box_loss += l_box
                total_cls_loss += l_cls
                total_obj_loss += l_obj

            # average over batch
            total_box_loss = total_box_loss / float(B)
            total_cls_loss = total_cls_loss / float(B)
            total_obj_loss = total_obj_loss / float(B)
            loss = self.lambda_box * total_box_loss + self.lambda_cls * total_cls_loss + self.lambda_obj * total_obj_loss
            return {'loss': loss, 'loss_components': {'box': total_box_loss, 'cls': total_cls_loss, 'obj': total_obj_loss}}
        else:
            # inference: return preds per-image list in same format [N, 5+C], preds are absolute xywh coords
            # convert back to per-image tensors
            return [preds_topk[i] for i in range(B)]


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # quick smoke test
    model = HSG_DET(num_classes=80,
                    channels=(256, 512, 1024),
                    sgb_k_p4=256, sgb_k_p5=256).eval()

    # synthetic batch for 1080p input (B,3,1080,1920)
    B = 2
    H, W = 1080, 1920
    imgs = torch.randn(B, 3, H, W)
    # forward inference (no targets)
    with torch.no_grad():
        preds = model(imgs, targets=None, epoch=0, total_epochs=300, topk_candidates=1024)
    print("Preds per image shapes:", [p.shape for p in preds])

    # synthetic targets for training step example
    model.train()
    sample_targets = []
    for b in range(B):
        # one GT box per image in xyxy pixel coords
        gt_box = torch.tensor([[400.0, 200.0, 800.0, 600.0]], dtype=torch.float32)
        gt_label = torch.tensor([1], dtype=torch.long)
        sample_targets.append({'boxes': gt_box, 'labels': gt_label})
    out = model(imgs, targets=sample_targets, epoch=10, total_epochs=300, topk_candidates=1024)
    print("Loss dict keys:", out.keys())
    print("Loss value:", out['loss'].item())