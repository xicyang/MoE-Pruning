"""Configuration for MoE Pruning Pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Default paths (can be configured via environment variables)
# Use current directory as default, or set via MOE_DATA_ROOT environment variable
DATA_ROOT = Path(os.getenv("MOE_DATA_ROOT", "."))
RESULT_BASE = Path(os.getenv("MOE_RESULT_BASE", str(DATA_ROOT / "pruneresult")))
MODEL_BASE = Path(os.getenv("MOE_MODEL_BASE", str(DATA_ROOT / "prunemodel")))

# Calibration datasets
CALIBRATION_DATASETS = ("arc", "winogrande", "medqa", "mmlu", "gsm8k", "hellaswag")


# Dataset paths (can be configured via environment variables)
# Users should set these via environment variables or provide full paths
# Example: export MOE_DATASET_MMLU=/path/to/mmlu
def _get_dataset_path(dataset_name: str, default_subpath: str) -> Path:
    """Get dataset path from environment variable or use default relative path."""
    env_key = f"MOE_DATASET_{dataset_name.upper()}"
    if env_key in os.environ:
        return Path(os.environ[env_key])
    return DATA_ROOT / default_subpath

DATASET_PATHS: Dict[str, Path] = {
    "medqa": _get_dataset_path("medqa", "datasets/GBaker/MedQA-USMLE-4-options-hf"),
    "mmlu": _get_dataset_path("mmlu", "datasets/cais/mmlu/all"),
    "arc": _get_dataset_path("arc", "datasets/allenai/ai2_arc/ARC-Challenge"),
    "winogrande": _get_dataset_path("winogrande", "datasets/allenai/winogrande/winogrande_xl"),
    "hellaswag": _get_dataset_path("hellaswag", "datasets/hellaswag/data"),
    "gsm8k": _get_dataset_path("gsm8k", "datasets/gsm8k/main"),
}

# Model aliases (can be configured via environment variables)
# Users should set these via environment variables or provide full paths
# Example: export MOE_MODEL_YOUR_MODEL=/path/to/your-model
def _get_model_alias(alias_name: str, default_subpath: str) -> Path:
    """Get model path from environment variable or use default relative path."""
    env_key = f"MOE_MODEL_{alias_name.upper().replace('-', '_').replace('@', '')}"
    if env_key in os.environ:
        return Path(os.environ[env_key])
    return DATA_ROOT / default_subpath

MODEL_ALIASES: Dict[str, Path] = {
    "Mixtral-8x7B-Instruct-v0.1": _get_model_alias("Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1"),
    "Mixtral-8x7B-v0.1": _get_model_alias("Mixtral-8x7B-v0.1", "Mixtral-8x7B-v0.1"),
    "@Mixtral-8x7B-Instruct-v0.1": _get_model_alias("Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1"),
    "@Mixtral-8x7B-v0.1": _get_model_alias("Mixtral-8x7B-v0.1", "Mixtral-8x7B-v0.1"),
}


@dataclass
class AnalysisConfig:
    """Configuration for expert analysis step."""
    model_path: Path
    output_base: Path
    dataset: str
    n_s: int
    k: int = 5
    max_block_size: int = 2048
    seed: int = 42
    split: str = "train"
    limit: int | None = None
    path_topk: int = 10
    path_limit: int = 25
    dtype: str = "bfloat16"
    cuda_devices: str = "0,1,2,3"
    extra_metadata: Dict[str, str] = field(default_factory=dict)
    merge_all_samples: bool = False  # Whether to merge all representative samples into one sample


@dataclass
class PathSearchConfig:
    """Configuration for path search step."""
    target_keep: int
    topk: int
    limit: int
    tau: float = 1.0
    loss_iqr_multiplier: float = 3.0
    norm_iqr_multiplier: float = 3.0
    min_values: int = 4
    unique_paths: bool = False
    enable_outlier_detection: bool = True


@dataclass
class PruningConfig:
    """Configuration for pruning step."""
    output_base: Path
    n_s: int
    verify: bool = True
    router_logit_value: float = -1e20


@dataclass
class WorkflowPaths:
    """Paths for workflow outputs."""
    analysis_dir: Path
    pruned_model_dir: Path

    @classmethod
    def derive(
        cls,
        model_name: str,
        dataset: str,
        result_base: Path | None = None,
        model_base: Path | None = None,
    ) -> "WorkflowPaths":
        """Derive workflow paths.
        
        Args:
            model_name: Name of the model
            dataset: Name of the dataset
            result_base: Base directory for analysis results (default: RESULT_BASE)
            model_base: Base directory for pruned models (default: MODEL_BASE)
        """
        result_base = result_base or RESULT_BASE
        model_base = model_base or MODEL_BASE
        analysis_dir = result_base / model_name / dataset / "analyze"
        pruned = model_base / model_name / dataset
        analysis_dir.mkdir(parents=True, exist_ok=True)
        pruned.mkdir(parents=True, exist_ok=True)
        return cls(analysis_dir=analysis_dir, pruned_model_dir=pruned)


def resolve_model_path(identifier: str | Path) -> Path:
    """Resolve model path from alias or literal path."""
    if isinstance(identifier, Path):
        return identifier
    normalized = identifier.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:]
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    return Path(identifier).expanduser()


def normalize_analysis_limit(user_value: int | None) -> int | None:
    """
    Interpret user-provided analysis sample limits.

    Returns:
        - None when user passes 0 or negative value (meaning "no limit").
        - The original positive integer for explicit caps.
        - Default limit when user does not specify.
    """
    DEFAULT_ANALYSIS_LIMIT = 0
    if user_value is None:
        return DEFAULT_ANALYSIS_LIMIT
    if user_value <= 0:
        return None
    return user_value
