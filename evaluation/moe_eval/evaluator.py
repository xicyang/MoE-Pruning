"""Model evaluation using lm-evaluation-harness with gate.bias support."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from moe_eval.config import EvalConfig
from moe_eval.load_bias_fix import load_gate_bias_from_checkpoint

logger = logging.getLogger(__name__)


def get_lm_eval_script_path() -> Path:
    """Get path to lm_eval_with_bias_fix.py script."""
    # Get the script path relative to this file
    script_path = Path(__file__).parent / "lm_eval_with_bias_fix.py"
    return script_path


def run_evaluation(
    model_path: Path,
    eval_tasks: List[str],
    output_dir: Path,
    cuda_devices: str = "0,1,2,3",
    hf_endpoint: Optional[str] = "https://hf-mirror.com",
    batch_size: int = 8,
    python_exec: Optional[str] = None,
    model_args: Optional[str] = None,
) -> None:
    """Run evaluation using lm-evaluation-harness with gate.bias support.

    Args:
        model_path: Path to the model to evaluate
        eval_tasks: List of evaluation tasks
        output_dir: Directory to save evaluation results
        cuda_devices: CUDA devices to use
        hf_endpoint: HuggingFace endpoint for model downloads
        batch_size: Batch size for evaluation
        python_exec: Python executable to use for evaluation
        model_args: Additional model arguments (e.g., "trust_remote_code=True")
    """
    logger.info(f"Starting evaluation for model: {model_path}")
    logger.info(f"Evaluation tasks: {eval_tasks}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Note: Using lm_eval_with_bias_fix.py to automatically load gate.bias for pruned models")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Set HF endpoint if provided
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    # Get the lm_eval_with_bias_fix script path
    lm_eval_script = get_lm_eval_script_path()
    if not lm_eval_script.exists():
        logger.error(f"lm_eval_with_bias_fix.py not found at {lm_eval_script}")
        raise FileNotFoundError(f"lm_eval_with_bias_fix.py not found at {lm_eval_script}")

    python_exec = python_exec or sys.executable
    logger.info(f"Using Python interpreter for evaluation: {python_exec}")

    # Build model_args string
    model_args_parts = [
        f"pretrained={model_path}",
        "parallelize=True",
    ]
    if model_args:
        model_args_parts.append(model_args)
    model_args_str = ",".join(model_args_parts)

    # Run evaluation for each task
    for task in eval_tasks:
        logger.info(f"Evaluating task: {task}")
        task_output_dir = output_dir / task
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Use lm_eval_with_bias_fix.py instead of lm_eval command
        # This ensures gate.bias is automatically loaded for pruned models
        cmd = [
            python_exec,
            "-m",
            "moe_eval.lm_eval_with_bias_fix",
            "--model",
            "hf",
            "--model_args",
            model_args_str,
            "--tasks",
            task,
            "--device",
            "cuda",
            "--batch_size",
            str(batch_size),
            "--output_path",
            str(task_output_dir),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Task {task} completed successfully")
            logger.debug(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Task {task} failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

    logger.info("Evaluation complete!")


def run_evaluation_with_config(config: EvalConfig) -> None:
    """Run evaluation using EvalConfig.

    Args:
        config: Evaluation configuration
    """
    run_evaluation(
        model_path=config.output_dir.parent,  # In our workflow, model is in parent of output
        eval_tasks=config.eval_tasks,
        output_dir=config.output_dir,
        cuda_devices=config.cuda_devices,
        hf_endpoint=config.hf_endpoint,
        batch_size=config.batch_size,
        python_exec=config.python_exec,
        model_args=config.model_args,
    )
