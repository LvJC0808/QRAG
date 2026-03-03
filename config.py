from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
QRAG_DIR = ROOT_DIR / "QRAG"
DATA_DIR = QRAG_DIR / "data"
INDEX_DIR = DATA_DIR / "indexes"
CACHE_DIR = DATA_DIR / "cache"
PROMPTS_DIR = QRAG_DIR / "prompts"

MODEL_ROOT = Path(os.getenv("QRAG_MODEL_ROOT", ROOT_DIR / "models"))
EMBED_MODEL_PATH = Path(os.getenv("QRAG_EMBED_MODEL", MODEL_ROOT / "qwen3-vl-embedding-8b"))
RERANK_MODEL_PATH = Path(os.getenv("QRAG_RERANK_MODEL", MODEL_ROOT / "qwen3-vl-reranker-8b"))
GEN_MODEL_PATH = Path(os.getenv("QRAG_GEN_MODEL", MODEL_ROOT / "qwen3-vl"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(slots=True)
class RuntimeConfig:
    embed_device: str = os.getenv("QRAG_EMBED_DEVICE", "cuda:0")
    rerank_device: str = os.getenv("QRAG_RERANK_DEVICE", "cuda:1")
    gen_device: str = os.getenv("QRAG_GEN_DEVICE", "cuda:2")
    judge_device: str = os.getenv("QRAG_JUDGE_DEVICE", "cuda:3")
    torch_dtype: str = os.getenv("QRAG_TORCH_DTYPE", "bfloat16")
    attn_implementation: str = os.getenv("QRAG_ATTN_IMPL", "flash_attention_2")


@dataclass(slots=True)
class RetrievalDefaults:
    recall_top_k: int = _env_int("QRAG_RECALL_TOP_K", 24)
    rerank_top_n: int = _env_int("QRAG_RERANK_TOP_N", 8)
    evidence_top_m: int = _env_int("QRAG_EVIDENCE_TOP_M", 4)
    image_max_pixels: int = _env_int("QRAG_IMAGE_MAX_PIXELS", 1024 * 1024)


@dataclass(slots=True)
class GenerationDefaults:
    max_new_tokens: int = _env_int("QRAG_MAX_NEW_TOKENS", 1200)
    temperature: float = float(os.getenv("QRAG_TEMPERATURE", "0.2"))
    force_table: bool = os.getenv("QRAG_FORCE_TABLE", "1") == "1"


@dataclass(slots=True)
class JudgeDefaults:
    score_threshold: int = _env_int("QRAG_JUDGE_THRESHOLD", 80)
    citation_threshold: int = _env_int("QRAG_CITATION_THRESHOLD", 75)


def ensure_directories() -> None:
    for path in [DATA_DIR, INDEX_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)
