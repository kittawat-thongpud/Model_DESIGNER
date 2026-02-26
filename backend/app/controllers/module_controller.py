"""
Module Controller — CRUD for custom nn.Module blocks + module catalog.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

from ..schemas.module import SaveModuleRequest
from ..services import module_storage, module_registry

router = APIRouter(prefix="/api/modules", tags=["Modules"])


@router.get("/", summary="List all custom modules")
async def list_modules():
    return module_storage.list_modules()


@router.get("/catalog", summary="Full module catalog (builtin + custom)")
async def get_catalog():
    """Return all available modules for the Model Designer palette."""
    return module_registry.get_all_modules()


@router.get("/catalog/categories", summary="Module categories")
async def get_categories():
    return module_registry.get_categories()


@router.get("/catalog/{name}", summary="Get module info by name")
async def get_module_info(name: str):
    info = module_registry.get_module_info(name)
    if not info:
        raise HTTPException(404, f"Module not found: {name}")
    return info


@router.get("/{module_id}", summary="Load a custom module")
async def load_module(module_id: str):
    record = module_storage.load_module(module_id)
    if not record:
        raise HTTPException(404, f"Module not found: {module_id}")
    return record


@router.post("/", summary="Save a custom module")
async def save_module(req: SaveModuleRequest):
    record = module_storage.save_module(
        name=req.name,
        code=req.code,
        args=[a.model_dump() for a in req.args],
        category=req.category,
        description=req.description,
    )
    return {"module_id": record["module_id"], "message": "Module saved"}


@router.put("/{module_id}", summary="Update a custom module")
async def update_module(module_id: str, req: SaveModuleRequest):
    existing = module_storage.load_module(module_id)
    if not existing:
        raise HTTPException(404, f"Module not found: {module_id}")
    record = module_storage.save_module(
        name=req.name,
        code=req.code,
        args=[a.model_dump() for a in req.args],
        category=req.category,
        description=req.description,
        module_id=module_id,
    )
    return {"module_id": record["module_id"], "message": "Module updated"}


@router.delete("/{module_id}", summary="Delete a custom module")
async def delete_module(module_id: str):
    if not module_storage.delete_module(module_id):
        raise HTTPException(404, f"Module not found: {module_id}")
    return {"message": "Deleted"}


@router.post("/{module_id}/validate", summary="Validate module code")
async def validate_module(module_id: str):
    """Try to compile the module code and check for syntax errors."""
    record = module_storage.load_module(module_id)
    if not record:
        raise HTTPException(404, f"Module not found: {module_id}")

    code = record.get("code", "")
    try:
        compile(code, f"<module:{record['name']}>", "exec")
        return {"valid": True, "message": "Code compiles successfully"}
    except SyntaxError as e:
        return {"valid": False, "message": f"Syntax error at line {e.lineno}: {e.msg}"}
