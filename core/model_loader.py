from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def import_module_from_path(module_name: str, source_path: Path) -> ModuleType:
    if not source_path.exists():
        raise FileNotFoundError(f"Module script not found: {source_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(source_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {source_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
