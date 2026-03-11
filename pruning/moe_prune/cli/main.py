"""Command line interface for Mixtral pruning pipeline."""

import argparse
import os
from pathlib import Path

from moe_prune.config import CALIBRATION_DATASETS
from moe_prune.utils import get_logger
from moe_prune.workflows import run_freq_rank_workflow

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoE Pruning Pipeline - Pruning workflow only"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path or alias (e.g., /path/to/model or @model-alias)",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="0,1,2,3",
        help="CUDA devices (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--analysis-k",
        type=int,
        default=5,
        help="Clustering number k for analysis (default: 5)",
    )
    parser.add_argument(
        "--target-keep",
        type=int,
        default=128,
        help="Target number of experts to keep (default: 128)",
    )
    parser.add_argument(
        "--path-topk",
        type=int,
        default=10,
        help="Number of top paths per sample (default: 10)",
    )
    parser.add_argument(
        "--path-limit",
        type=int,
        default=25,
        help="Max samples to process in path search (default: 25)",
    )
    parser.add_argument(
        "--calibration-datasets",
        type=str,
        nargs="+",
        default=list(CALIBRATION_DATASETS),
        choices=list(CALIBRATION_DATASETS),
        help="Calibration datasets to use",
    )
    parser.add_argument(
        "--merge-all-samples",
        action="store_true",
        help="Merge all representative samples from all clusters into a single sample for path planning",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Base directory for analysis results (default: ./pruneresult or from MIXTRAL_RESULT_BASE env)",
    )
    parser.add_argument(
        "--pruned-model-dir",
        type=str,
        default=None,
        help="Directory to save pruned models (default: ./prunemodel/{model_name}/{dataset} or from MIXTRAL_MODEL_BASE env). "
             "If specified, this will be used directly (without appending model_name/dataset)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # n_S is fixed to 1
    n_s_values = [1]

    # Parse output paths
    output_base = Path(args.output_base) if args.output_base else None
    pruned_model_dir = Path(args.pruned_model_dir) if args.pruned_model_dir else None

    logger.info("=" * 80)
    logger.info("MoE Pruning Pipeline - Frequency Rank Workflow")
    logger.info("=" * 80)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"n_S: 1 (fixed)")
    logger.info(f"CUDA devices: {args.cuda_devices}")
    logger.info(f"Calibration datasets: {args.calibration_datasets}")
    if output_base:
        logger.info(f"Analysis results base: {output_base}")
    if pruned_model_dir:
        logger.info(f"Pruned models directory: {pruned_model_dir}")
    logger.info("=" * 80)

    run_freq_rank_workflow(
        model_path=args.model_path,
        calibration_datasets=args.calibration_datasets,
        cuda_devices=args.cuda_devices,
        analysis_k=args.analysis_k,
        target_keep=args.target_keep,
        path_topk=args.path_topk,
        path_limit=args.path_limit,
        merge_all_samples=args.merge_all_samples,
        output_base=output_base,
        pruned_model_dir=pruned_model_dir,
    )

    logger.info("=" * 80)
    logger.info("All workflows completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
