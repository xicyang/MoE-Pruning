"""Configuration for MoE Model Evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalConfig:
    eval_tasks: List[str]
    output_dir: Path
    cuda_devices: str = "0,1,2,3"
    hf_endpoint: Optional[str] = "https://hf-mirror.com"
    batch_size: int = 8
    python_exec: Optional[str] = None
    model_args: Optional[str] = None


DEFAULT_EVAL_TASKS = [
    "mmlu",
    "arc_challenge",
    "winogrande",
    "hellaswag",
    "gsm8k",
    "medqa_4options",
]