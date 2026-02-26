
import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from app.schemas.model_schema import ModelPackage, ModelGraph, PackageParameter, GlobalVariable

# Store packages in backend/data/packages
PACKAGE_DIR = Path(__file__).parent.parent.parent / "data" / "packages"
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

class PackageService:
    @staticmethod
    def list_packages() -> List[ModelPackage]:
        packages = []
        if not PACKAGE_DIR.exists():
            return []
            
        for f in PACKAGE_DIR.glob("*.pkg.json"):
            try:
                # Read valid JSONs
                with open(f, 'r') as file:
                    pkg = ModelPackage.model_validate_json(file.read())
                packages.append(pkg)
            except Exception as e:
                print(f"Error loading package {f}: {e}")
        return packages

    @staticmethod
    def get_package(package_id: str) -> Optional[ModelPackage]:
        path = PACKAGE_DIR / f"{package_id}.pkg.json"
        if path.exists():
            try:
                with open(path, 'r') as file:
                    return ModelPackage.model_validate_json(file.read())
            except Exception:
                return None
        return None

    @staticmethod
    def create_package(
        source_graph: ModelGraph, 
        name: str, 
        description: str,
        exposed_global_ids: List[str]
    ) -> ModelPackage:
        """
        Create a new package from a model graph.
        exposed_global_ids: List of GlobalVariable IDs to expose as package parameters.
        """
        # Generate ID from name
        pkg_id = name.lower().strip().replace(" ", "_")
        
        # Filter exposed globals to create parameters
        exposed_params = []
        relevant_globals = []
        
        # Map globals
        global_map = {g.id: g for g in source_graph.globals}
        
        for g_id in exposed_global_ids:
            if g_id in global_map:
                g = global_map[g_id]
                exposed_params.append(PackageParameter(
                    name=g.name,
                    type=g.type,
                    default=g.value,
                    description=g.description or f"Parameter for {g.name}",
                    options=g.options
                ))
                relevant_globals.append(g)
        
        # We store ALL globals in the package to ensure internal logic works,
        # but only 'exposed_params' are visible to the user of the package.
        # Alternatively, we could filter globals to only used ones, but that requires graph analysis.
        # For now, store all source globals to be safe.
        
        new_package = ModelPackage(
            id=pkg_id,
            name=name,
            description=description,
            nodes=source_graph.nodes,
            edges=source_graph.edges,
            globals=source_graph.globals,
            exposed_params=exposed_params,
            created_at=datetime.utcnow()
        )
        
        # Save to file
        file_path = PACKAGE_DIR / f"{pkg_id}.pkg.json"
        with open(file_path, "w") as f:
            f.write(new_package.model_dump_json(indent=2))
            
        return new_package

    @staticmethod
    def delete_package(package_id: str) -> bool:
        path = PACKAGE_DIR / f"{package_id}.pkg.json"
        if path.exists():
            path.unlink()
            return True
        return False
