from fastapi import APIRouter, HTTPException, Body
from typing import List

from ..schemas.model_schema import ModelPackage, CreatePackageRequest
from ..services.package_service import PackageService
from .. import logging_service as logger

router = APIRouter(prefix="/api/packages", tags=["Packages"])

@router.get("/", response_model=List[ModelPackage], summary="List all packages")
async def list_packages():
    """List all available model packages."""
    return PackageService.list_packages()

@router.get("/{package_id}", response_model=ModelPackage, summary="Get package details")
async def get_package(package_id: str):
    """Get a specific package by ID."""
    pkg = PackageService.get_package(package_id)
    if not pkg:
        raise HTTPException(status_code=404, detail="Package not found")
    return pkg

@router.post("/", response_model=ModelPackage, summary="Create/Export a package")
async def create_package(request: CreatePackageRequest):
    """
    Create a new package from the given model graph.
    Exposed globals will become the package's parameters.
    """
    try:
        pkg = PackageService.create_package(
            request.graph, 
            request.name, 
            request.description, 
            request.exposed_globals
        )
        logger.log("model", "INFO", f"Package created: {pkg.name}", {"package_id": pkg.id})
        return pkg
    except Exception as e:
        logger.log("model", "ERROR", f"Failed to create package: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{package_id}", summary="Delete a package")
async def delete_package(package_id: str):
    """Delete a package."""
    if PackageService.delete_package(package_id):
        logger.log("model", "INFO", f"Package deleted", {"package_id": package_id})
        return {"message": "Package deleted"}
    raise HTTPException(status_code=404, detail="Package not found")
