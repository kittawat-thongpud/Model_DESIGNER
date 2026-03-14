"""
Model DESIGNER MCP Server.

Exposes FastMCP tools and resources that wrap the existing backend
REST API surface for models, datasets, jobs, weights, benchmark, and
training control.

Mount via FastAPI:
    from .mcp.server import create_mcp_app
    app.mount("/mcp", create_mcp_app())
"""
from __future__ import annotations
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# ── Tool imports ─────────────────────────────────────────────────────────────
from .tools import models as _models
from .tools import datasets as _datasets
from .tools import jobs as _jobs
from .tools import weights as _weights
from .tools import benchmark as _benchmark
from .tools import training as _training

# ── Server instance ──────────────────────────────────────────────────────────

mcp = FastMCP(
    name="model-designer",
    instructions=(
        "Model DESIGNER MCP interface. Use these tools to manage models, "
        "datasets, training jobs, weights, and benchmarks. "
        "All list tools default to summary view to reduce token usage. "
        "Pass view='detail' to get full records."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
            "0.0.0.0:*",
            "10.46.136.183:*",
            "10.46.136.189:*",
        ],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
            "http://0.0.0.0:*",
            "http://10.46.136.183:*",
            "http://10.46.136.189:*",
        ],
    ),
)


# ════════════════════════════════════════════════════════════════════════════
# MODEL TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_models(
    task: str | None = None,
    view: str = "summary",
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """List all model definitions.

    Args:
        task: Filter by task type (detect, segment, classify, pose, obb).
        view: 'summary' (default) or 'detail'.
        limit: Max items to return.
        offset: Items to skip.
    """
    return _models.list_models(task=task, view=view, limit=limit, offset=offset)


@mcp.tool()
def get_model(model_id: str, view: str = "summary") -> dict[str, Any]:
    """Get a model by ID.

    Args:
        model_id: Target model ID.
        view: 'summary' (default) or 'detail' (includes yaml_def, graph metadata).
    """
    return _models.get_model(model_id=model_id, view=view)


@mcp.tool()
def get_model_yaml(model_id: str) -> dict[str, Any]:
    """Get the raw YAML string for a model.

    Args:
        model_id: Target model ID.
    """
    return _models.get_model_yaml(model_id=model_id)


@mcp.tool()
def create_model(
    name: str,
    task: str,
    yaml_def: dict,
    description: str = "",
    replace: bool = False,
) -> dict[str, Any]:
    """Create or replace a model definition.

    Args:
        name: Human-readable model name.
        task: Task type (detect, segment, classify, pose, obb).
        yaml_def: Ultralytics-compatible YAML definition as dict.
        description: Optional description.
        replace: If True, overwrite an existing model with the same name.
    """
    return _models.create_model(
        name=name, task=task, yaml_def=yaml_def,
        description=description, replace=replace,
    )


@mcp.tool()
def validate_model(model_id: str, scale: str | None = None) -> dict[str, Any]:
    """Validate a model YAML with Ultralytics and return param/layer counts.

    Args:
        model_id: Target model ID.
        scale: Optional scale char (n, s, m, l, x).
    """
    return _models.validate_model(model_id=model_id, scale=scale)


# ════════════════════════════════════════════════════════════════════════════
# DATASET TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_datasets(
    view: str = "summary",
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """List all available datasets.

    Args:
        view: 'summary' (default) or 'detail'.
        limit: Max items to return.
        offset: Items to skip.
    """
    return _datasets.list_datasets(view=view, limit=limit, offset=offset)


@mcp.tool()
def get_dataset(name: str, view: str = "summary") -> dict[str, Any]:
    """Get dataset info by name.

    Args:
        name: Dataset name (e.g. 'coco128', 'idd').
        view: 'summary' (default) or 'detail'.
    """
    return _datasets.get_dataset(name=name, view=view)


@mcp.tool()
def preview_dataset(name: str, count: int = 4) -> dict[str, Any]:
    """Get sample entries from a dataset (labels only, no base64 images).

    Args:
        name: Dataset name.
        count: Number of samples (default 4, max 16).
    """
    return _datasets.preview_dataset(name=name, count=count)


# ════════════════════════════════════════════════════════════════════════════
# JOB TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_jobs(
    status: str | None = None,
    model_id: str | None = None,
    view: str = "summary",
    limit: int | None = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List training jobs with optional filters.

    Args:
        status: Filter by status (pending, running, completed, failed, stopped).
        model_id: Filter by model ID.
        view: 'summary' (default) or 'detail'.
        limit: Max items to return (default 50).
        offset: Items to skip.
    """
    return _jobs.list_jobs(
        status=status, model_id=model_id, view=view, limit=limit, offset=offset,
    )


@mcp.tool()
def get_job(
    job_id: str,
    include_history: bool = False,
    view: str = "summary",
) -> dict[str, Any]:
    """Get a training job record.

    Args:
        job_id: Target job ID.
        include_history: Include epoch-by-epoch history (increases response size).
        view: 'summary' (default) or 'detail'.
    """
    return _jobs.get_job(job_id=job_id, include_history=include_history, view=view)


@mcp.tool()
def get_job_logs(
    job_id: str,
    limit: int = 50,
    offset: int = 0,
    level: str | None = None,
) -> dict[str, Any]:
    """Get training logs for a job (most recent first).

    Args:
        job_id: Target job ID.
        limit: Max log lines (default 50).
        offset: Lines to skip.
        level: Filter by log level (INFO, WARNING, ERROR, DEBUG, PROGRESS).
    """
    return _jobs.get_job_logs(job_id=job_id, limit=limit, offset=offset, level=level)


@mcp.tool()
def get_job_metrics(job_id: str) -> dict[str, Any]:
    """Get current system metrics (GPU, CPU, RAM) for a job.

    Args:
        job_id: Target job ID.
    """
    return _jobs.get_job_metrics(job_id=job_id)


@mcp.tool()
def get_job_history(
    job_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Get epoch-by-epoch training metrics history.

    Args:
        job_id: Target job ID.
        limit: Max epochs to return.
        offset: Epochs to skip.
    """
    return _jobs.get_job_history(job_id=job_id, limit=limit, offset=offset)


# ════════════════════════════════════════════════════════════════════════════
# WEIGHT TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_weights(
    model_id: str | None = None,
    view: str = "summary",
    limit: int | None = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List all saved weights.

    Args:
        model_id: Filter by parent model ID.
        view: 'summary' (default) or 'detail'.
        limit: Max items to return (default 50).
        offset: Items to skip.
    """
    return _weights.list_weights(model_id=model_id, view=view, limit=limit, offset=offset)


@mcp.tool()
def get_weight(weight_id: str, view: str = "summary") -> dict[str, Any]:
    """Get weight metadata by ID.

    Args:
        weight_id: Target weight ID.
        view: 'summary' (default) or 'detail' (includes training_runs, edits).
    """
    return _weights.get_weight(weight_id=weight_id, view=view)


@mcp.tool()
def get_weight_info(weight_id: str) -> dict[str, Any]:
    """Get model param count and GFLOPs from a weight .pt file.

    Args:
        weight_id: Target weight ID.
    """
    return _weights.get_weight_info(weight_id=weight_id)


@mcp.tool()
def get_weight_lineage(weight_id: str, view: str = "summary") -> dict[str, Any]:
    """Get the training lineage chain for a weight (oldest ancestor first).

    Args:
        weight_id: Target weight ID.
        view: 'summary' (default) or 'detail'.
    """
    return _weights.get_weight_lineage(weight_id=weight_id, view=view)


@mcp.tool()
def create_empty_weight(
    model_id: str | None = None,
    yolo_model: str | None = None,
    use_pretrained: bool = False,
    model_scale: str | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Create an empty weight from a model architecture or official YOLO checkpoint.

    Args:
        model_id: Custom model ID. Mutually exclusive with yolo_model.
        yolo_model: Official YOLO key e.g. 'yolov8n', 'yolov8s'.
        use_pretrained: Load COCO-pretrained weights; False = random init.
        model_scale: Scale char (n, s, m, l, x) for custom models.
        name: Optional display name.
    """
    return _weights.create_empty_weight(
        model_id=model_id,
        yolo_model=yolo_model,
        use_pretrained=use_pretrained,
        model_scale=model_scale,
        name=name,
    )


@mcp.tool()
def list_pretrained_weights(
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """List available pretrained model catalog from all weight source plugins.

    Args:
        limit: Max items to return.
        offset: Items to skip.
    """
    return _weights.list_pretrained_weights(limit=limit, offset=offset)


@mcp.tool()
def download_pretrained_weight(model_key: str) -> dict[str, Any]:
    """Download a pretrained model and save as a weight record.

    Args:
        model_key: Pretrained model key (e.g. 'yolov8n', 'yolov8s-seg').
    """
    return _weights.download_pretrained_weight(model_key=model_key)


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARK TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_benchmark_datasets(weight_id: str | None = None) -> dict[str, Any]:
    """List datasets available for benchmarking.

    Args:
        weight_id: Optional weight to filter compatible datasets.
    """
    return _benchmark.list_benchmark_datasets(weight_id=weight_id)


@mcp.tool()
def run_benchmark(
    weight_id: str,
    dataset: str,
    split: str = "val",
    conf: float = 0.001,
    iou: float = 0.6,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
) -> dict[str, Any]:
    """Run benchmark validation for a weight. Returns mAP, latency, and per-class metrics.

    Args:
        weight_id: Target weight ID.
        dataset: Dataset name.
        split: Dataset split (val, test, train).
        conf: Confidence threshold.
        iou: IoU threshold.
        imgsz: Inference image size.
        batch: Batch size.
        device: Device string ('' = auto, 'cpu', '0', 'cuda:0').
    """
    return _benchmark.run_benchmark(
        weight_id=weight_id, dataset=dataset, split=split,
        conf=conf, iou=iou, imgsz=imgsz, batch=batch, device=device,
    )


@mcp.tool()
def list_benchmarks(
    weight_id: str | None = None,
    view: str = "summary",
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List past benchmark results.

    Args:
        weight_id: Filter by weight ID.
        view: 'summary' (default) or 'detail'.
        limit: Max items to return (default 20).
        offset: Items to skip.
    """
    return _benchmark.list_benchmarks(
        weight_id=weight_id, view=view, limit=limit, offset=offset,
    )


@mcp.tool()
def get_benchmark(benchmark_id: str, view: str = "detail") -> dict[str, Any]:
    """Get a specific benchmark result by ID.

    Args:
        benchmark_id: Benchmark ID.
        view: 'summary' or 'detail' (default).
    """
    return _benchmark.get_benchmark(benchmark_id=benchmark_id, view=view)


# ════════════════════════════════════════════════════════════════════════════
# TRAINING CONTROL TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def start_training(
    model_id: str,
    config: dict,
    model_scale: str | None = None,
    partitions: list[dict] | None = None,
) -> dict[str, Any]:
    """Start a training job. Returns job_id; training runs in background queue.

    Args:
        model_id: Model ID (use 'yolo:yolov8n' for official YOLO models).
        config: Training config (required: 'data'; common: epochs, batch, imgsz).
        model_scale: Scale char (n, s, m, l, x) for custom models.
        partitions: List of partition split configs
                    [{'partition_id': 'p_xxx', 'train': true, 'val': false}].
    """
    return _training.start_training(
        model_id=model_id, config=config,
        model_scale=model_scale, partitions=partitions,
    )


@mcp.tool()
def stop_training(job_id: str) -> dict[str, Any]:
    """Signal a running training job to stop.

    Args:
        job_id: Target job ID.
    """
    return _training.stop_training(job_id=job_id)


@mcp.tool()
def resume_training(job_id: str) -> dict[str, Any]:
    """Resume a stopped or failed training job from its last checkpoint.

    Args:
        job_id: Target job ID.
    """
    return _training.resume_training(job_id=job_id)


@mcp.tool()
def append_training(job_id: str, additional_epochs: int = 50) -> dict[str, Any]:
    """Append additional epochs to a completed or stopped training job.

    Args:
        job_id: Target job ID.
        additional_epochs: Extra epochs to train (default 50).
    """
    return _training.append_training(job_id=job_id, additional_epochs=additional_epochs)


@mcp.tool()
def get_training_queue() -> dict[str, Any]:
    """Get current training queue status including pending jobs and slot limits."""
    return _training.get_training_queue()


@mcp.tool()
def get_training_workers_health() -> dict[str, Any]:
    """Get health status of all active training worker threads."""
    return _training.get_training_workers_health()


# ════════════════════════════════════════════════════════════════════════════
# MCP RESOURCES
# ════════════════════════════════════════════════════════════════════════════

@mcp.resource("models://list")
def resource_models_list() -> str:
    """All models — summary list."""
    import json
    result = _models.list_models(view="summary")
    return json.dumps(result, ensure_ascii=False)


@mcp.resource("models://{model_id}")
def resource_model(model_id: str) -> str:
    """Model record by ID — summary."""
    import json
    return json.dumps(_models.get_model(model_id, view="summary"), ensure_ascii=False)


@mcp.resource("models://{model_id}/yaml")
def resource_model_yaml(model_id: str) -> str:
    """Raw YAML for a model."""
    import json
    return json.dumps(_models.get_model_yaml(model_id), ensure_ascii=False)


@mcp.resource("datasets://list")
def resource_datasets_list() -> str:
    """All datasets — summary list."""
    import json
    return json.dumps(_datasets.list_datasets(view="summary"), ensure_ascii=False)


@mcp.resource("datasets://{name}")
def resource_dataset(name: str) -> str:
    """Dataset info by name — summary."""
    import json
    return json.dumps(_datasets.get_dataset(name, view="summary"), ensure_ascii=False)


@mcp.resource("jobs://list")
def resource_jobs_list() -> str:
    """All jobs — summary list (newest 50)."""
    import json
    return json.dumps(_jobs.list_jobs(view="summary", limit=50), ensure_ascii=False)


@mcp.resource("jobs://{job_id}")
def resource_job(job_id: str) -> str:
    """Training job record by ID — summary."""
    import json
    return json.dumps(_jobs.get_job(job_id, view="summary"), ensure_ascii=False)


@mcp.resource("jobs://{job_id}/logs")
def resource_job_logs(job_id: str) -> str:
    """Recent training logs for a job (last 50 lines)."""
    import json
    return json.dumps(_jobs.get_job_logs(job_id, limit=50), ensure_ascii=False)


@mcp.resource("jobs://{job_id}/metrics")
def resource_job_metrics(job_id: str) -> str:
    """System metrics for a job."""
    import json
    return json.dumps(_jobs.get_job_metrics(job_id), ensure_ascii=False)


@mcp.resource("jobs://{job_id}/history")
def resource_job_history(job_id: str) -> str:
    """Epoch-by-epoch training history for a job."""
    import json
    return json.dumps(_jobs.get_job_history(job_id), ensure_ascii=False)


@mcp.resource("weights://list")
def resource_weights_list() -> str:
    """All weights — summary list (newest 50)."""
    import json
    return json.dumps(_weights.list_weights(view="summary", limit=50), ensure_ascii=False)


@mcp.resource("weights://{weight_id}")
def resource_weight(weight_id: str) -> str:
    """Weight metadata by ID — summary."""
    import json
    return json.dumps(_weights.get_weight(weight_id, view="summary"), ensure_ascii=False)


@mcp.resource("weights://{weight_id}/lineage")
def resource_weight_lineage(weight_id: str) -> str:
    """Training lineage chain for a weight."""
    import json
    return json.dumps(_weights.get_weight_lineage(weight_id, view="summary"), ensure_ascii=False)


@mcp.resource("benchmarks://list")
def resource_benchmarks_list() -> str:
    """All benchmarks — summary list (newest 20)."""
    import json
    return json.dumps(_benchmark.list_benchmarks(view="summary", limit=20), ensure_ascii=False)


@mcp.resource("benchmarks://{benchmark_id}")
def resource_benchmark(benchmark_id: str) -> str:
    """Benchmark result by ID — detail."""
    import json
    return json.dumps(_benchmark.get_benchmark(benchmark_id, view="detail"), ensure_ascii=False)


@mcp.resource("training://queue")
def resource_training_queue() -> str:
    """Current training queue status."""
    import json
    return json.dumps(_training.get_training_queue(), ensure_ascii=False)


@mcp.resource("training://workers/health")
def resource_training_workers_health() -> str:
    """Health status of active training workers."""
    import json
    return json.dumps(_training.get_training_workers_health(), ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════════════
# FASTAPI MOUNT HELPER
# ════════════════════════════════════════════════════════════════════════════

def create_mcp_app():
    """Return a Starlette/ASGI app for mounting in FastAPI under /mcp.

    Exposes two endpoints:
      GET  /mcp/sse       — SSE stream (MCP client connects here)
      POST /mcp/messages  — JSON-RPC message posting
    """
    return mcp.sse_app(mount_path="/")
