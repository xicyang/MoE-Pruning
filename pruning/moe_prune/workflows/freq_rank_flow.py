"""Freq-rank workflow for Mixtral pruning."""

from pathlib import Path
from typing import List

from moe_prune.analysis import run_analysis
from moe_prune.config import (
    AnalysisConfig,
    PathSearchConfig,
    PruningConfig,
    WorkflowPaths,
    normalize_analysis_limit,
    resolve_model_path,
)
from moe_prune.pruning import apply_pruning_mask, run_path_search
from moe_prune.utils import get_logger

logger = get_logger(__name__)


def run_freq_rank_workflow(
    *,
    model_path: str | Path,
    calibration_datasets: List[str],
    cuda_devices: str,
    analysis_k: int = 5,
    target_keep: int = 128,
    path_topk: int = 10,
    path_limit: int = 25,
    analysis_limit: int | None = None,
    merge_all_samples: bool = False,
    output_base: Path | None = None,
    pruned_model_dir: Path | None = None,
) -> None:
    n_s = 1

    for dataset in calibration_datasets:
        if pruned_model_dir is not None:
            pruned_dir = pruned_model_dir
            pruned_dir.mkdir(parents=True, exist_ok=True)
            paths = WorkflowPaths(
                analysis_dir=WorkflowPaths.derive(model_name, dataset, result_base=output_base).analysis_dir,
                pruned_model_dir=pruned_dir,
            )
        else:
            paths = WorkflowPaths.derive(model_name, dataset, result_base=output_base)

        logger.info(
            f"freq-rank workflow: model={model_name} dataset={dataset} n_S={n_s}"
        )

        # Analysis
        logger.info("Step 1: Analysis")
        analysis = run_analysis(
            AnalysisConfig(
                model_path=model_path,
                output_base=paths.analysis_dir,
                dataset=dataset,
                n_s=n_s,
                k=analysis_k,
                cuda_devices=cuda_devices,
                path_topk=path_topk,
                path_limit=path_limit,
                limit=normalize_analysis_limit(analysis_limit),
                extra_metadata={"workflow": "freq_rank"},
                merge_all_samples=merge_all_samples,
            )
        )

        # Path search
        logger.info("Step 2: Path Search")
        path_result = run_path_search(
            analysis.output_dir,
            config=PathSearchConfig(
                target_keep=target_keep,
                topk=path_topk,
                limit=path_limit,
            ),
        )

        # Pruning
        logger.info("Step 3: Pruning")
        prune_result = apply_pruning_mask(
            model_path=model_path,
            mask_path=path_result.mask_path,
            config=PruningConfig(
                output_base=paths.pruned_model_dir,
                n_s=n_s,
            ),
        )

        # Write experiment config
        _write_experiment_config(
            analysis.output_dir,
            model_name=model_name,
            dataset=dataset,
            n_s=n_s,
            analysis_dir=analysis.output_dir,
            mask_path=path_result.mask_path,
            pruned_model=prune_result.pruned_model_dir,
            analysis_k=analysis_k,
            target_keep=target_keep,
            path_topk=path_topk,
            path_limit=path_limit,
        )

    logger.info(f"freq-rank workflow completed for model={model_name}")


def _write_experiment_config(
    output_dir: Path,
    *,
    model_name: str,
    dataset: str,
    n_s: int,
    analysis_dir: Path,
    mask_path: Path,
    pruned_model: Path,
    analysis_k: int,
    target_keep: int,
    path_topk: int,
    path_limit: int,
) -> None:
    content = f"""实验配置 (freq-rank, n_S={n_s}, 数据集={dataset})
==================
原始模型: {model_name}
校准数据集: {dataset}
n_S: {n_s}
K(聚类数): {analysis_k}

步骤1输出目录: {analysis_dir}
步骤2掩码文件: {mask_path}
步骤3剪枝模型: {pruned_model}

剪枝参数:
  target_keep: {target_keep}
  topk: {path_topk}
  limit: {path_limit}
"""
    with (output_dir / "experiment_config.txt").open("w", encoding="utf-8") as f:
        f.write(content)