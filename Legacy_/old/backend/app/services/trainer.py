"""
Training engine — runs PyTorch training in a background thread.
Applies full training configuration: optimizer selection, LR scheduling,
warmup, early stopping, AMP, data augmentation, layer freezing, seeding.
"""
from __future__ import annotations
import threading
import time
import uuid
import traceback
import math
from datetime import datetime
from typing import Any
import warnings
try:
    from numpy import VisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass

from .. import logging_service as logger
from ..services import job_storage
from ..services import weight_storage

# Dataset class names for analysis
_CLASS_NAMES: dict[str, list[str]] = {
    "mnist": [str(i) for i in range(10)],
    "fashion_mnist": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}

# In-memory job cache (hydrated from disk on access)
_jobs: dict[str, dict[str, Any]] = {}
_stop_flags: dict[str, threading.Event] = {}


def _create_job(model_id: str, model_name: str, config: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    record = {
        "job_id": job_id,
        "model_id": model_id,
        "model_name": model_name,
        "config": config,
        "status": "pending",
        "epoch": 0,
        "total_epochs": config.get("epochs", 5),
        "train_loss": 0.0,
        "train_accuracy": 0.0,
        "val_loss": None,
        "val_accuracy": None,
        "message": "Queued",
        "history": [],
        "weight_id": None,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
    }
    _jobs[job_id] = record
    _stop_flags[job_id] = threading.Event()
    job_storage.save_job(record)
    job_storage.append_job_log(job_id, "INFO", "Job created", {"model_id": model_id, "config": config})
    return job_id


def _detection_collate_fn(batch):
    """Custom collate for detection (handles variable number of boxes)."""
    return tuple(zip(*batch))


def _build_transforms(config: dict, dataset_name: str):
    """Build torchvision transforms based on config. Only add augmentations for image datasets."""
    from torchvision import transforms

    # Determine if dataset is image-based
    image_datasets = {"mnist", "cifar10", "cifar100", "fashion_mnist", "svhn", "coco"}
    is_image = dataset_name.lower() in image_datasets

    # Determine normalization based on dataset
    if dataset_name.lower() in ("cifar10", "cifar100", "svhn"):
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        normalize = transforms.Normalize((0.5,), (0.5,))

    # Image size resize (0 = use native)
    imgsz = config.get("imgsz", 0)

    # ── Train transforms ──
    train_tfms = []

    if imgsz > 0 and is_image:
        train_tfms.append(transforms.Resize((imgsz, imgsz)))

    if is_image:
        # Random crop
        crop_frac = config.get("crop_fraction", 1.0)
        if crop_frac < 1.0 and imgsz > 0:
            train_tfms.append(transforms.RandomResizedCrop(imgsz, scale=(crop_frac, 1.0)))

        # Geometric augmentations
        degrees = config.get("degrees", 0.0)
        translate = config.get("translate", 0.0)
        scale = config.get("scale", 0.0)
        shear = config.get("shear", 0.0)
        if degrees > 0 or translate > 0 or scale > 0 or shear > 0:
            scale_range = (1.0 - scale, 1.0 + scale) if scale > 0 else None
            translate_range = (translate, translate) if translate > 0 else None
            shear_range = (-shear, shear) if shear > 0 else None
            train_tfms.append(transforms.RandomAffine(
                degrees=degrees,
                translate=translate_range,
                scale=scale_range,
                shear=shear_range,
            ))

        # Flip augmentations
        fliplr = config.get("fliplr", 0.0)
        if fliplr > 0:
            train_tfms.append(transforms.RandomHorizontalFlip(p=fliplr))
        flipud = config.get("flipud", 0.0)
        if flipud > 0:
            train_tfms.append(transforms.RandomVerticalFlip(p=flipud))

        # Color augmentations
        hsv_h = config.get("hsv_h", 0.0)
        hsv_s = config.get("hsv_s", 0.0)
        hsv_v = config.get("hsv_v", 0.0)
        if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
            train_tfms.append(transforms.ColorJitter(
                brightness=hsv_v if hsv_v > 0 else 0,
                saturation=hsv_s if hsv_s > 0 else 0,
                hue=min(hsv_h, 0.5) if hsv_h > 0 else 0,
            ))

        # Auto augment
        auto_aug = config.get("auto_augment", "")
        if auto_aug == "randaugment":
            train_tfms.append(transforms.RandAugment())
        elif auto_aug == "autoaugment":
            train_tfms.append(transforms.AutoAugment())
        elif auto_aug == "trivialaugmentwide":
            train_tfms.append(transforms.TrivialAugmentWide())

    train_tfms.append(transforms.ToTensor())
    train_tfms.append(normalize)

    # Random erasing (must be after ToTensor)
    if is_image:
        erasing = config.get("erasing", 0.0)
        if erasing > 0:
            train_tfms.append(transforms.RandomErasing(p=erasing))

    # ── Val transforms (no augmentation) ──
    val_tfms = []
    if imgsz > 0 and is_image:
        val_tfms.append(transforms.Resize((imgsz, imgsz)))
    val_tfms.append(transforms.ToTensor())
    val_tfms.append(normalize)

    return transforms.Compose(train_tfms), transforms.Compose(val_tfms)


def _build_optimizer(model, config: dict):
    """Create optimizer based on config."""
    import torch.optim as optim

    opt_name = config.get("optimizer", "Adam")
    lr = config.get("lr0", config.get("learning_rate", 0.001))
    wd = config.get("weight_decay", 0.0005)
    mom = config.get("momentum", 0.9)

    if opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    elif opt_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(mom, 0.999))
    else:  # Adam
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(mom, 0.999))


def _build_scheduler(optimizer, config: dict):
    """Create LR scheduler based on config."""
    import torch.optim.lr_scheduler as sched

    epochs = config.get("epochs", 5)
    cos_lr = config.get("cos_lr", False)
    lrf = config.get("lrf", 0.01)

    if cos_lr:
        return sched.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=optimizer.defaults['lr'] * lrf)
    else:
        # StepLR: decay by lrf factor every 1/3 of total epochs
        step = max(1, epochs // 3)
        gamma = lrf ** (1.0 / max(1, epochs // step))
        return sched.StepLR(optimizer, step_size=step, gamma=gamma)


def _apply_warmup(optimizer, config: dict, epoch: int, batch_idx: int, num_batches: int):
    """Apply linear warmup for learning rate and momentum."""
    warmup_epochs = config.get("warmup_epochs", 0)
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return

    lr0 = config.get("lr0", config.get("learning_rate", 0.001))
    warmup_bias_lr = config.get("warmup_bias_lr", 0.1)
    warmup_momentum = config.get("warmup_momentum", 0.8)
    target_momentum = config.get("momentum", 0.9)

    # Progress through warmup (0 → 1)
    total_warmup_iters = warmup_epochs * num_batches
    current_iter = (epoch - 1) * num_batches + batch_idx
    progress = min(1.0, current_iter / max(1, total_warmup_iters))

    for pg in optimizer.param_groups:
        # Linear warmup for LR
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg['lr']
        base_lr = warmup_bias_lr if pg.get('is_bias', False) else 0.0
        pg['lr'] = base_lr + (lr0 - base_lr) * progress

        # Linear warmup for momentum
        if 'momentum' in pg:
            pg['momentum'] = warmup_momentum + (target_momentum - warmup_momentum) * progress
        elif 'betas' in pg:
            pg['betas'] = (warmup_momentum + (target_momentum - warmup_momentum) * progress, pg['betas'][1])


def _training_worker(job_id: str, model_id: str, config: dict):
    """Background training loop with persistent logging."""
    job = _jobs[job_id]
    stop_flag = _stop_flags[job_id]

    try:
        import torch
        import torch.nn as nn
        from torchvision import datasets

        # ── Seeding ──
        seed = config.get("seed", 0)
        if seed > 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            import random, numpy as np
            random.seed(seed)
            np.random.seed(seed)

        deterministic = config.get("deterministic", False)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ── Load model from saved graph ──
        from ..storage import load_model, WEIGHTS_DIR
        from .codegen import generate_code

        _, graph = load_model(model_id)
        class_name, code = generate_code(graph)

        namespace: dict = {}
        exec(code, namespace)
        ModelClass = namespace[class_name]
        
        # Apply global config overrides
        global_overrides = config.get("global_overrides", {})
        model = ModelClass(**global_overrides)

        # ── Load pretrained weights if specified ──
        pretrained_id = config.get("pretrained", "")
        if pretrained_id:
            weight_path = WEIGHTS_DIR / f"{pretrained_id}.pt"
            if weight_path.exists():
                state_dict = torch.load(str(weight_path), map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                job_storage.append_job_log(job_id, "INFO", f"Loaded pretrained weights: {pretrained_id}")

        # ── Device ──
        device_str = config.get("device", "auto")
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)
        model = model.to(device)

        # ── Freeze layers ──
        freeze_n = config.get("freeze", 0)
        if freeze_n > 0:
            layers = list(model.children())
            for i, layer in enumerate(layers[:freeze_n]):
                for param in layer.parameters():
                    param.requires_grad = False
            job_storage.append_job_log(job_id, "INFO", f"Frozen first {freeze_n} layers")

        # ── Dataset & Transforms ──
        dataset_name = config.get("dataset", "mnist").lower()
        train_transform, val_transform = _build_transforms(config, dataset_name)

        if dataset_name == "mnist":
            train_ds = datasets.MNIST("./data", train=True, download=True, transform=train_transform)
            val_ds = datasets.MNIST("./data", train=False, download=True, transform=val_transform)
        elif dataset_name == "cifar10":
            train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
            val_ds = datasets.CIFAR10("./data", train=False, download=True, transform=val_transform)
        elif dataset_name == "fashion_mnist":
            train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=train_transform)
            val_ds = datasets.FashionMNIST("./data", train=False, download=True, transform=val_transform)
        elif dataset_name == "coco":
            # COCO requires manual download usually, but we assume it's mounted or available
            # We use a try-except to give a helpful error
            try:
                # Use CocoDetection (requires pycocotools)
                # We expect standard structure: ./data/coco/{train2017, val2017, annotations}
                train_ds = datasets.CocoDetection(
                    root="./data/coco/train2017",
                    annFile="./data/coco/annotations/instances_train2017.json",
                    transform=train_transform
                )
                val_ds = datasets.CocoDetection(
                    root="./data/coco/val2017",
                    annFile="./data/coco/annotations/instances_val2017.json",
                    transform=val_transform
                )
            except Exception as e:
                raise RuntimeError(f"COCO dataset not found or pycocotools missing: {e}")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        batch_size = config.get("batch_size", 64)
        workers = config.get("workers", 2)
        
        collate = _detection_collate_fn if dataset_name == "coco" else None
        
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate
        )

        # ── Loss ──
        cls_weight = config.get("cls_weight", 1.0)
        
        task_type = "classification"
        if dataset_name == "coco":
            task_type = "detection"
            from . import detection_utils
            criterion = detection_utils.DetectionLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # ── Optimizer & Scheduler ──
        optimizer = _build_optimizer(model, config)
        scheduler = _build_scheduler(optimizer, config)

        # ── AMP ──
        use_amp = config.get("amp", False) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ── Training loop ──
        epochs = config.get("epochs", 5)
        do_val = config.get("val", True)
        patience = config.get("patience", 0)
        save_period = config.get("save_period", 0)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        all_preds: list = []
        all_targets: list = []

        job["status"] = "running"
        job["total_epochs"] = epochs
        job["message"] = "Training started"
        job["started_at"] = datetime.utcnow().isoformat() + "Z"
        job_storage.save_job(job)

        config_summary = {
            "device": str(device), "optimizer": config.get("optimizer", "Adam"),
            "lr0": config.get("lr0", 0.001), "amp": use_amp,
            "augmentation": any([
                config.get("degrees", 0) > 0, config.get("fliplr", 0) > 0,
                config.get("hsv_h", 0) > 0, config.get("erasing", 0) > 0,
                config.get("auto_augment", "") != "",
            ]),
        }
        job_storage.append_job_log(job_id, "INFO", "Training started", config_summary)
        logger.log("training", "INFO", f"Training started for model {model_id}", {"job_id": job_id, **config_summary})

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            if stop_flag.is_set():
                job["status"] = "stopped"
                job["message"] = f"Stopped at epoch {epoch}"
                job["completed_at"] = datetime.utcnow().isoformat() + "Z"
                job_storage.save_job(job)
                job_storage.append_job_log(job_id, "INFO", f"Training stopped at epoch {epoch}")
                logger.log("training", "INFO", f"Training stopped at epoch {epoch}", {"job_id": job_id})
                return

            # ── Train epoch ──
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            num_batches = len(train_loader)

            # Gradient accumulation: simulate larger effective batch via NBS
            nbs = config.get("nbs", 64)
            accumulate = max(1, round(nbs / batch_size))  # accumulation steps
            optimizer.zero_grad()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if stop_flag.is_set():
                    break

                # Apply warmup
                _apply_warmup(optimizer, config, epoch, batch_idx, num_batches)

                # Prepare inputs
                inputs = inputs.to(device)
                
                # For classification, targets are tensors. For detection, they are lists (collate).
                # We encode detection targets after getting model output to know grid size S.
                if task_type != "detection":
                    targets = targets.to(device)

                if use_amp and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        if task_type == "detection":
                            # outputs: (B, 5+C, S, S)
                            S = outputs.shape[2]
                            num_classes = outputs.shape[1] - 5
                            targets_tensor = detection_utils.encode_target(targets, num_classes, S, device)
                            loss = criterion(outputs, targets_tensor) * cls_weight / accumulate
                        else:
                            loss = criterion(outputs, targets) * cls_weight / accumulate
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % accumulate == 0 or (batch_idx + 1) == num_batches:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    outputs = model(inputs)
                    if task_type == "detection":
                         S = outputs.shape[2]
                         num_classes = outputs.shape[1] - 5
                         targets_tensor = detection_utils.encode_target(targets, num_classes, S, device)
                         loss = criterion(outputs, targets_tensor) * cls_weight / accumulate
                    else:
                        loss = criterion(outputs, targets) * cls_weight / accumulate
                            
                    loss.backward()
                    if (batch_idx + 1) % accumulate == 0 or (batch_idx + 1) == num_batches:
                        optimizer.step()
                        optimizer.zero_grad()

                running_loss += loss.item() * accumulate  # undo scaling for logging
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss = running_loss / max(num_batches, 1)
            train_acc = 100.0 * correct / max(total, 1)

            # ── Validate ──
            val_loss = None
            val_acc = None
            epoch_preds: list = []
            epoch_targets: list = []

            if do_val:
                model.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device)
                        if task_type != "detection":
                            targets = targets.to(device)
                            
                        if use_amp:
                            with torch.amp.autocast("cuda"):
                                outputs = model(inputs)
                                if task_type == "detection":
                                    S = outputs.shape[2]
                                    num_classes = outputs.shape[1] - 5
                                    t_tensor = detection_utils.encode_target(targets, num_classes, S, device)
                                    loss = criterion(outputs, t_tensor)
                                else:
                                    loss = criterion(outputs, targets)
                        else:
                            outputs = model(inputs)
                            if task_type == "detection":
                                S = outputs.shape[2]
                                num_classes = outputs.shape[1] - 5
                                t_tensor = detection_utils.encode_target(targets, num_classes, S, device)
                                loss = criterion(outputs, t_tensor)
                            else:
                                loss = criterion(outputs, targets)
                                
                        v_loss += loss.item()
                        
                        if task_type != "detection":
                            _, predicted = outputs.max(1)
                            v_total += targets.size(0)
                            v_correct += predicted.eq(targets).sum().item()
                            epoch_preds.extend(predicted.cpu().tolist())
                            epoch_targets.extend(targets.cpu().tolist())

                val_loss = v_loss / max(len(val_loader), 1)
                val_acc = 100.0 * v_correct / max(v_total, 1)

            # Accumulate for final confusion matrix
            all_preds = epoch_preds
            all_targets = epoch_targets

            # ── Compute classification metrics ──
            ep_precision = None
            ep_recall = None
            ep_f1 = None
            if epoch_preds and epoch_targets:
                try:
                    from sklearn.metrics import precision_recall_fscore_support
                    p, r, f, _ = precision_recall_fscore_support(
                        epoch_targets, epoch_preds, average="macro", zero_division=0
                    )
                    ep_precision = round(float(p), 4)
                    ep_recall = round(float(r), 4)
                    ep_f1 = round(float(f), 4)
                except Exception:
                    pass

            # Step scheduler (after warmup period)
            warmup_epochs = config.get("warmup_epochs", 0)
            if epoch > warmup_epochs:
                scheduler.step()

            # ── Timing & GPU memory ──
            epoch_time = round(time.time() - epoch_start, 2)
            current_lr = optimizer.param_groups[0]['lr']
            gpu_mem = None
            if device.type == "cuda":
                gpu_mem = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)
                torch.cuda.reset_peak_memory_stats(device)

            # ── Track best values ──
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc

            # ── Update job state ──
            job["epoch"] = epoch
            job["train_loss"] = round(train_loss, 4)
            job["train_accuracy"] = round(train_acc, 2)
            job["val_loss"] = round(val_loss, 4) if val_loss is not None else None
            job["val_accuracy"] = round(val_acc, 2) if val_acc is not None else None
            job["best_val_loss"] = round(best_val_loss, 4) if best_val_loss < float("inf") else None
            job["best_val_accuracy"] = round(best_val_acc, 2) if best_val_acc > 0 else None
            job["message"] = f"Epoch {epoch}/{epochs} complete"
            job["history"].append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_cls_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 2),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "val_cls_loss": round(val_loss, 4) if val_loss is not None else None,
                "val_accuracy": round(val_acc, 2) if val_acc is not None else None,
                "precision": ep_precision,
                "recall": ep_recall,
                "f1": ep_f1,
                "lr": round(current_lr, 8),
                "epoch_time": epoch_time,
                "gpu_memory_mb": gpu_mem,
            })

            # Persist
            job_storage.save_job(job)
            log_msg = f"Epoch {epoch}/{epochs}: loss={train_loss:.4f} acc={train_acc:.2f}%"
            if val_loss is not None:
                log_msg += f" val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
            if ep_precision is not None:
                log_msg += f" P={ep_precision:.3f} R={ep_recall:.3f} F1={ep_f1:.3f}"
            log_msg += f" lr={current_lr:.6f} {epoch_time}s"
            if gpu_mem is not None:
                log_msg += f" GPU={gpu_mem}MB"
            job_storage.append_job_log(
                job_id, "INFO", log_msg,
                {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                 "val_loss": val_loss, "val_accuracy": val_acc, "lr": current_lr,
                 "precision": ep_precision, "recall": ep_recall, "f1": ep_f1,
                 "epoch_time": epoch_time, "gpu_memory_mb": gpu_mem}
            )
            logger.log("training", "INFO", log_msg, {"job_id": job_id})

            # ── Save periodic checkpoint ──
            if save_period > 0 and epoch % save_period == 0:
                from ..storage import WEIGHTS_DIR
                ckpt_id = f"{job_id}_ep{epoch}"
                ckpt_path = WEIGHTS_DIR / f"{ckpt_id}.pt"
                torch.save(model.state_dict(), str(ckpt_path))
                job_storage.append_job_log(job_id, "INFO", f"Checkpoint saved: epoch {epoch}", {"weight_id": ckpt_id})

            # ── Weight & Activation snapshot recording ──
            wr_enabled = config.get("weight_record_enabled", False)
            wr_freq = config.get("weight_record_frequency", 5)
            wr_layers = config.get("weight_record_layers", [])
            
            if wr_enabled and epoch % wr_freq == 0:
                try:
                    from ..services import weight_snapshots
                    recorded_weights = 0
                    recorded_acts = 0

                    # 1. Record Weights (Parameters)
                    # We check if the prefix "node_X" matches wr_layers
                    for name, param in model.named_parameters():
                        # name is e.g. "node_1.weight"
                        layer_id = name.split('.')[0]
                        if not wr_layers or (layer_id in wr_layers):
                            weight_snapshots.save_snapshot(job_id, epoch, name, param.data)
                            recorded_weights += 1
                    
                    # 2. Record Activations (Outputs) via Hook
                    # We need a batch input to run forward pass
                    inputs_example = None
                    try:
                        # Grab one batch from validation loader (or train if val empty)
                        iter_loader = iter(val_loader) if do_val else iter(train_loader)
                        inputs_example, _ = next(iter_loader)
                        inputs_example = inputs_example.to(device)
                    except StopIteration:
                        pass
                        
                    if inputs_example is not None:
                        activations = {}
                        hooks = []
                        
                        def get_activation(name):
                            def hook(model, input, output):
                                activations[name] = output.detach()
                            return hook
                            
                        # Register hooks on selected layers
                        for name, module in model.named_children():
                             # name is like "node_1"
                             if not wr_layers or (name in wr_layers):
                                 h = module.register_forward_hook(get_activation(name))
                                 hooks.append(h)
                                 
                        if hooks:
                            # Run forward pass in eval mode
                            was_training = model.training
                            model.eval()
                            with torch.no_grad():
                                if use_amp and scaler is not None:
                                    with torch.amp.autocast("cuda"):
                                        model(inputs_example)
                                else:
                                    model(inputs_example)
                            model.train(was_training)
                                    
                            # Remove hooks
                            for h in hooks:
                                h.remove()
                                
                            # Save activations
                            for name, tensor in activations.items():
                                # Save as "node_1.output"
                                weight_snapshots.save_snapshot(job_id, epoch, f"{name}.output", tensor)
                                recorded_acts += 1

                    job_storage.append_job_log(
                        job_id, "INFO",
                        f"Snapshot: epoch {epoch}, params={recorded_weights}, acts={recorded_acts}"
                    )
                except Exception as e:
                    job_storage.append_job_log(
                        job_id, "WARNING", f"Snapshot failed: {e}"
                    )

            # ── Early stopping ──
            if patience > 0 and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        job_storage.append_job_log(job_id, "INFO",
                            f"Early stopping at epoch {epoch} (patience={patience})")
                        logger.log("training", "INFO",
                            f"Early stopping at epoch {epoch}", {"job_id": job_id})
                        break

        # ── Compute final analysis ──
        dataset_name_lower = config.get("dataset", "mnist").lower()
        class_names = _CLASS_NAMES.get(dataset_name_lower, [])
        job["class_names"] = class_names

        if all_preds and all_targets:
            try:
                from sklearn.metrics import confusion_matrix as cm_func
                from sklearn.metrics import precision_recall_fscore_support

                cm = cm_func(all_targets, all_preds)
                job["confusion_matrix"] = cm.tolist()

                p, r, f, sup = precision_recall_fscore_support(
                    all_targets, all_preds, average=None, zero_division=0
                )
                per_class = []
                for i in range(len(p)):
                    name = class_names[i] if i < len(class_names) else str(i)
                    per_class.append({
                        "class": name,
                        "precision": round(float(p[i]), 4),
                        "recall": round(float(r[i]), 4),
                        "f1": round(float(f[i]), 4),
                        "support": int(sup[i]),
                    })
                job["per_class_metrics"] = per_class
            except Exception as e:
                job_storage.append_job_log(job_id, "WARNING", f"Could not compute analysis: {e}")

        # ── Save final weights ──
        from ..storage import WEIGHTS_DIR
        weight_id = uuid.uuid4().hex[:12]
        weight_path = WEIGHTS_DIR / f"{weight_id}.pt"
        torch.save(model.state_dict(), str(weight_path))

        weight_storage.save_weight_meta(
            model_id=model_id,
            model_name=job.get("model_name", "Untitled"),
            job_id=job_id,
            dataset=config.get("dataset", "mnist"),
            epochs_trained=job["epoch"],
            final_accuracy=job.get("val_accuracy"),
            final_loss=job.get("val_loss"),
            weight_id=weight_id,
        )

        job["status"] = "completed"
        job["message"] = "Training complete"
        job["weight_id"] = weight_id
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job_storage.save_job(job)
        job_storage.append_job_log(job_id, "INFO", "Training complete. Weights saved.", {"weight_id": weight_id})
        logger.log("training", "INFO", f"Training complete for model {model_id}. Weight: {weight_id}",
                    {"job_id": job_id, "weight_id": weight_id})

    except Exception as e:
        job["status"] = "failed"
        job["message"] = str(e)
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job_storage.save_job(job)
        job_storage.append_job_log(job_id, "ERROR", f"Training failed: {e}", {"traceback": traceback.format_exc()})
        logger.log("training", "ERROR", f"Training failed: {e}", {"job_id": job_id, "traceback": traceback.format_exc()})


def start_training(model_id: str, model_name: str, config: dict) -> str:
    """Start a training job in a background thread. Returns job_id."""
    job_id = _create_job(model_id, model_name, config)
    thread = threading.Thread(target=_training_worker, args=(job_id, model_id, config), daemon=True)
    thread.start()
    return job_id


def get_job_status(job_id: str) -> dict | None:
    if job_id in _jobs:
        return _jobs[job_id]
    return job_storage.load_job(job_id)


def stop_training(job_id: str) -> bool:
    if job_id in _stop_flags:
        _stop_flags[job_id].set()
        return True
    return False
