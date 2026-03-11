"""Command line interface for Mixtral evaluation."""

import argparse
import logging
import os
import sys
from pathlib import Path

from moe_eval.config import EvalConfig, DEFAULT_EVAL_TASKS
from moe_eval.evaluator import run_evaluation

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoE Model Evaluation - Evaluate models with lm-evaluation-harness"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model to evaluate",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=DEFAULT_EVAL_TASKS,
        help="Evaluation tasks (default: all common tasks)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="0,1,2,3",
        help="CUDA devices (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HuggingFace endpoint (default: https://hf-mirror.com)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable to use for evaluation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("MoE Model Evaluation Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Evaluation tasks: {args.tasks}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"CUDA devices: {args.cuda_devices}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 80)

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    run_evaluation(
        model_path=model_path,
        eval_tasks=args.tasks,
        output_dir=output_dir,
        cuda_devices=args.cuda_devices,
        hf_endpoint=args.hf_endpoint,
        batch_size=args.batch_size,
        python_exec=args.python,
        model_args="trust_remote_code=True",
    )

    logger.info("=" * 80)
    logger.info("Evaluation completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
