"""
Model Controller — CRUD + build for model graphs.
Tagged as "Models" for ReDoc grouping.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..schemas.model_schema import ModelGraph, ModelSummary
from .. import storage
from ..services.codegen import generate_code
from ..services import build_storage
from .. import logging_service as logger

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.post("/", response_model=dict, summary="Save a model graph")
async def save_model(graph: ModelGraph, replace: bool = False):
    """Save or create a new model. Handles overwrite if name exists."""
    # 1. Update existing by ID
    if graph.id:
        # Check collision with other models
        existing = storage.find_model_by_name(graph.meta.name)
        if existing and existing["id"] != graph.id:
            if not replace:
                return {
                    "exists": True,
                    "model_id": existing["id"],
                    "name": existing["name"],
                    "message": "Model with this name already exists"
                }
            # Overwrite: delete the other model first
            storage.delete_model(existing["id"])
            
        model_id = storage.save_model(graph, graph.id)
        logger.log("model", "INFO", f"Model updated: {graph.meta.name}", {"model_id": model_id})
        return {"model_id": model_id, "message": "Model updated successfully"}

    # 2. Check for name collision
    existing = storage.find_model_by_name(graph.meta.name)
    if existing:
        if not replace:
            return {
                "exists": True,
                "model_id": existing["id"],
                "name": existing["name"],
                "message": "Model with this name already exists"
            }
        # Overwrite existing
        model_id = storage.save_model(graph, existing["id"])
        logger.log("model", "INFO", f"Model overwritten: {graph.meta.name}", {"model_id": model_id})
        return {"model_id": model_id, "message": "Model overwritten successfully"}

    # 3. Create new
    model_id = storage.save_model(graph)
    logger.log("model", "INFO", f"Model created: {graph.meta.name}", {"model_id": model_id})
    return {"model_id": model_id, "message": "Model created successfully"}


@router.get("/", response_model=list[ModelSummary], summary="List all saved models")
async def list_models():
    """Return a summary of all saved models."""
    return storage.list_models()


@router.get("/{model_id}", response_model=ModelGraph, summary="Load a model")
async def get_model(model_id: str):
    """Load a specific model graph by its ID."""
    try:
        _, graph = storage.load_model(model_id)
        return graph
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.delete("/{model_id}", summary="Delete a model")
async def delete_model(model_id: str):
    """Delete a saved model."""
    if storage.delete_model(model_id):
        logger.log("model", "INFO", f"Model deleted", {"model_id": model_id})
        return {"message": "Model deleted"}
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@router.post("/{model_id}/build", summary="Generate PyTorch code and persist build")
async def build_model(model_id: str, replace: bool = False):
    """Generate PyTorch nn.Module code from a saved model graph and save build record.

    If a build with the same model name already exists and `replace` is not set,
    returns `{ exists: true, existing_build_id }` so the frontend can ask for confirmation.
    """
    try:
        _, graph = storage.load_model(model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    class_name, code = generate_code(graph)

    # Check for duplicate name
    existing = build_storage.find_build_by_name(graph.meta.name)
    if existing and not replace:
        return {
            "exists": True,
            "existing_build_id": existing["build_id"],
            "model_name": graph.meta.name,
            "model_id": model_id,
            "code": code,
            "class_name": class_name,
        }

    # If replacing, delete the old build
    if existing and replace:
        build_storage.delete_build(existing["build_id"])

    # Extract layer info from graph nodes
    layers = []
    for i, node in enumerate(graph.nodes):
        if node.type not in ("Input", "Output"):
            layers.append({
                "index": i,
                "layer_type": node.type,
                "params": node.params,
            })

    from datetime import datetime
    build_record = {
        "model_id": model_id,
        "model_name": graph.meta.name,
        "class_name": class_name,
        "code": code,
        "layers": layers,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    build_id = build_storage.save_build(build_record)

    logger.log("model", "INFO", f"Code generated and build saved for {class_name}", {"model_id": model_id, "build_id": build_id})
    return {
        "exists": False,
        "build_id": build_id,
        "model_id": model_id,
        "code": code,
        "class_name": class_name,
    }
