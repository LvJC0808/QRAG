"""QRAG package for local multimodal PDF RAG."""

from __future__ import annotations

from typing import Any


def create_pipeline(*args: Any, **kwargs: Any):
    # Lazy import keeps lightweight commands (e.g. batch dry-run) from triggering CUDA init.
    from .pipeline_factory import create_pipeline as _create_pipeline

    return _create_pipeline(*args, **kwargs)


__all__ = ["create_pipeline"]
