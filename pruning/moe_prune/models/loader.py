"""Model loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelBundle:
    """Container for model, tokenizer, and config."""
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: AutoConfig
    name: str

    def to(self, device: torch.device):
        self.model.to(device)
        return self


def load_model_bundle(
    model_path: Path,
    *,
    dtype: str = "bfloat16",
    device_map: Optional[str] = "auto",
    use_flash_attention_2: bool = False,
) -> ModelBundle:
    """Load model, tokenizer, and config."""
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    torch_dtype = _resolve_dtype(dtype)

    # Setup max_memory for multi-GPU
    max_memory_dict = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            usable_mem = int(total_mem * 0.8)
            max_memory_dict[i] = f"{usable_mem}GB"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=max_memory_dict if max_memory_dict else None,
        attn_implementation="flash_attention_2" if use_flash_attention_2 else None,
    )

    # Try fast tokenizer first, fallback to slow tokenizer if it fails
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        name=model_path.name,
    )


def _resolve_dtype(dtype: str) -> torch.dtype:
    """Resolve dtype string to torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = dtype.lower()
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]
