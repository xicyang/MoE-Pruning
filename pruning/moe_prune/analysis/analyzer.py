"""Expert analysis module."""

import json
import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from moe_prune.config import AnalysisConfig, DATASET_PATHS
from moe_prune.datasets import load_multi_domain_datasets, format_dataset_texts
from moe_prune.models.expert_analyzer import ExpertAnalyzerWrapper
from moe_prune.models.loader import load_model_bundle
from moe_prune.pruning.pruner import get_moe_block
from moe_prune.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    output_dir: Path
    model_name: str


def build_hashed_bow_features(tokenized_texts: List[List[int]], hash_dim: int = 1024) -> np.ndarray:
    features = np.zeros((len(tokenized_texts), hash_dim), dtype=np.float32)
    for i, ids in enumerate(tokenized_texts):
        for tid in ids:
            bucket = int(tid) % hash_dim
            features[i, bucket] += 1.0
        norm = np.linalg.norm(features[i])
        if norm > 0:
            features[i] /= norm
    return features


def kmeans(
    features: np.ndarray, k: int, seed: int = 42, max_iters: int = 100, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    assert features.ndim == 2
    n, d = features.shape
    rng = np.random.RandomState(seed)
    assert k <= n, "k must be <= n"

    # k-means++ init
    centers = np.empty((k, d), dtype=np.float32)
    idx0 = rng.randint(0, n)
    centers[0] = features[idx0]
    closest_dist_sq = np.sum((features - centers[0]) ** 2, axis=1)

    for ci in range(1, k):
        probs = closest_dist_sq / np.sum(closest_dist_sq)
        r = rng.rand()
        cum = np.cumsum(probs)
        next_idx = np.searchsorted(cum, r)
        centers[ci] = features[next_idx]
        dist_sq_new_center = np.sum((features - centers[ci]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_new_center)

    labels = np.zeros(n, dtype=np.int32)
    for it in range(max_iters):
        dists = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        counts = np.bincount(new_labels, minlength=k).astype(np.float32)
        for ci in range(k):
            if counts[ci] > 0:
                new_centers[ci] = features[new_labels == ci].mean(axis=0)
            else:
                ridx = rng.randint(0, n)
                new_centers[ci] = features[ridx]
                counts[ci] = 1.0
        shift = np.linalg.norm(new_centers - centers)
        centers, labels = new_centers, new_labels
        if shift < tol:
            break
    return centers, labels


def select_n_per_cluster(
    labels: np.ndarray, features: np.ndarray, centers: np.ndarray, n_s: int
) -> List[List[int]]:
    k = centers.shape[0]
    selected: List[List[int]] = []
    for ci in range(k):
        idxs = np.where(labels == ci)[0]
        if len(idxs) == 0:
            selected.append([])
            continue
        sub = features[idxs]
        d2 = ((sub - centers[ci][None, :]) ** 2).sum(axis=1)
        n_select = min(n_s, len(idxs))
        sorted_indices = np.argsort(d2)[:n_select]
        selected.append(idxs[sorted_indices].tolist())
    return selected


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> str:
    inputs = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer.strip()


def concatenate_samples(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_block_size: int,
    eos_token_id: int | None = None,
) -> Dict[str, torch.Tensor]:
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    all_input_ids = []
    for i, text in enumerate(texts):
        add_special = i == 0
        encoded = tokenizer(text, add_special_tokens=add_special, truncation=False)
        input_ids = encoded["input_ids"]
        all_input_ids.append(input_ids)

    concatenated = []
    for i, input_ids in enumerate(all_input_ids):
        if i > 0 and eos_token_id is not None:
            concatenated.append(eos_token_id)
        concatenated.extend(input_ids)

    if len(concatenated) > max_block_size:
        concatenated = concatenated[:max_block_size]

    seq_len = len(concatenated)
    attention_mask = [1] * seq_len
    labels = list(concatenated)

    return {
        "input_ids": torch.tensor(concatenated, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_dataset_kmeans_samples(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    dataset_type: str,
    dataset_paths: Dict[str, Path],
    split: str,
    k: int,
    n_s: int,
    max_block_size: int,
    seed: int,
    use_cache: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    set_seed(seed)

    raw_items = load_multi_domain_datasets(
        dataset_to_load=dataset_type,
        mmlu_path=str(dataset_paths.get("mmlu", DATASET_PATHS["mmlu"])),
        arc_path=str(dataset_paths.get("arc", DATASET_PATHS["arc"])),
        medqa_path=str(dataset_paths.get("medqa", DATASET_PATHS["medqa"])),
        winogrande_path=str(dataset_paths.get("winogrande", DATASET_PATHS["winogrande"])),
        hellaswag_path=str(dataset_paths.get("hellaswag", DATASET_PATHS["hellaswag"])),
        gsm8k_path=str(dataset_paths.get("gsm8k", DATASET_PATHS["gsm8k"])),
        split=split,
    )

    if len(raw_items) == 0:
        raise RuntimeError(f"{dataset_type.upper()} data is empty")

    question_texts = format_dataset_texts(raw_items, dataset_type=dataset_type, question_only=True)
    cap = max(4 * k * n_s, k * n_s)
    if len(question_texts) > cap:
        indices = list(range(len(question_texts)))
        random.Random(seed).shuffle(indices)
        indices = indices[:cap]
        question_texts = [question_texts[i] for i in indices]
        raw_items = [raw_items[i] for i in indices]

    enc = tokenizer(question_texts, add_special_tokens=True, truncation=True, max_length=max_block_size)
    token_lists: List[List[int]] = enc["input_ids"]
    feats = build_hashed_bow_features(token_lists, hash_dim=1024)
    centers, labels = kmeans(feats, k=k, seed=seed)
    chosen_indices_per_cluster = select_n_per_cluster(labels, feats, centers, n_s=n_s)

    logger.info(f"Generating calibration data for {len(chosen_indices_per_cluster)} clusters...")

    samples: List[Dict[str, torch.Tensor]] = []

    for cluster_idx, indices in enumerate(tqdm(chosen_indices_per_cluster, desc="Processing clusters")):
        if len(indices) == 0:
            logger.warning(f"Cluster {cluster_idx} has no samples, skipping")
            continue

        cluster_texts_with_answers = []
        for idx in indices:
            question = raw_items[idx].get("question", "")
            try:
                generated_answer = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    use_cache=use_cache,
                )
            except Exception as e:
                logger.warning(f"Cluster {cluster_idx}, sample {idx}: Failed to generate answer: {e}, using original")
                generated_answer = raw_items[idx].get("answer", "")

            qa_text = f"Question: {question}\nAnswer: {generated_answer}"
            cluster_texts_with_answers.append(qa_text)

        concatenated_sample = concatenate_samples(
            tokenizer=tokenizer,
            texts=cluster_texts_with_answers,
            max_block_size=max_block_size,
            eos_token_id=tokenizer.eos_token_id,
        )
        samples.append(concatenated_sample)

    return samples


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    logger.info(f"Starting analysis with config: {config}")

    model_path = Path(config.model_path)
    if str(model_path).endswith("/"):
        model_path = Path(str(model_path)[:-1])

    model_name = model_path.name
    save_path = config.output_base / f"{model_name}_expert_analysis_{config.dataset}_k{config.k}_nS{config.n_s}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_path.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    max_block_size = min(config.max_block_size, 2048)

    logger.info("Loading model and tokenizer...")
    bundle = load_model_bundle(
        model_path,
        dtype=config.dtype,
        device_map="auto",
        use_flash_attention_2=False,
    )
    model = bundle.model
    tokenizer = bundle.tokenizer

    logger.info(f"Model device map: {model.hf_device_map}")

    total_layers = len(model.model.layers)
    logger.info(f"Total transformer layers: {total_layers}")

    # Wrap MoE layers
    wrapped_layers: Dict[int, ExpertAnalyzerWrapper] = {}
    moe_layer_indices: List[int] = []
    for layer_idx in range(total_layers):
        layer = model.model.layers[layer_idx]
        try:
            moe_block = get_moe_block(layer)
        except ValueError as exc:
            logger.warning(f"Layer {layer_idx}: {exc}, skipping.")
            continue

        has_experts = hasattr(moe_block, "experts") and len(getattr(moe_block, "experts", [])) > 0
        has_gate = hasattr(moe_block, "gate")
        if not (has_experts and has_gate):
            logger.info(f"Layer {layer_idx}: not an MoE block, skipping.")
            continue

        logical_idx = len(moe_layer_indices)
        moe_layer_indices.append(layer_idx)
        wrapped = ExpertAnalyzerWrapper(moe_block)
        wrapped_layers[logical_idx] = wrapped
        if hasattr(layer, "block_sparse_moe"):
            layer.block_sparse_moe = wrapped
        elif hasattr(layer, "mlp"):
            layer.mlp = wrapped

    if not moe_layer_indices:
        raise RuntimeError("No MoE layers detected in the model; nothing to analyze.")

    logger.info(f"Detected {len(moe_layer_indices)} MoE layers: {moe_layer_indices}")

    layer_map = {
        "model_name": model_name,
        "total_layers": total_layers,
        "moe_layer_indices": moe_layer_indices,
        "moe_layer_count": len(moe_layer_indices),
        "created_at": datetime.now().isoformat(),
    }
    mapping_path = save_path / "moe_layer_map.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(layer_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved MoE layer map to {mapping_path}")

    # Build samples
    logger.info(f"Building representative samples from {config.dataset.upper()} + k-means...")
    samples = build_dataset_kmeans_samples(
        tokenizer=tokenizer,
        model=model,
        dataset_type=config.dataset,
        dataset_paths=DATASET_PATHS,
        split=config.split,
        k=config.k,
        n_s=config.n_s,
        max_block_size=max_block_size,
        seed=config.seed,
        use_cache=True,
    )
    logger.info(f"Selected {len(samples)} large samples (each concatenated from {config.n_s} sub-samples)")

    device = model.model.embed_tokens.weight.device
    logger.info(f"Input device: {device}")

    # Forward pass for each sample
    with torch.inference_mode():
        for sample_idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
            for logical_idx in range(len(wrapped_layers)):
                wrapped_layers[logical_idx].enable_recording = True
                wrapped_layers[logical_idx].clear_records()

            sample_on_device = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    v = v.unsqueeze(0).to(device)
                sample_on_device[k] = v

            sample_on_device["use_cache"] = False
            _ = model(**sample_on_device)
            torch.cuda.empty_cache()

            # Compute expert loss
            per_sample_results: Dict[int, Dict[str, Any]] = {}
            for logical_idx, actual_layer_idx in enumerate(moe_layer_indices):
                wrapped = wrapped_layers[logical_idx]
                if len(wrapped.all_inputs) == 0:
                    logger.warning(
                        f"Sample {sample_idx}: Logical layer {logical_idx} (model layer {actual_layer_idx}) recorded no data, skipping"
                    )
                    continue
                try:
                    loss_scores = wrapped.compute_expert_loss()
                    stats = wrapped.get_statistics()
                    layer_results = {
                        "layer_idx": logical_idx,
                        "model_layer_idx": actual_layer_idx,
                        "loss": loss_scores,
                        "router_probs_mean": stats["router_probs_mean"].tolist(),
                        "expert_norms_mean": stats["expert_norms_mean"].tolist(),
                        "sample_count": int(stats["sample_count"]),
                    }
                    per_sample_results[logical_idx] = layer_results
                except Exception as e:
                    logger.error(
                        f"Sample {sample_idx}: Logical layer {logical_idx} (model layer {actual_layer_idx}) computation failed: {e}",
                        exc_info=True,
                    )
                finally:
                    torch.cuda.empty_cache()

            # Save JSON
            json_results: Dict[str, Any] = {}
            for logical_idx, layer_data in per_sample_results.items():
                json_results[f"layer_{logical_idx}"] = {
                    "loss": {f"expert_{k}": float(v) for k, v in layer_data["loss"].items()},
                    "router_probs_mean": [float(x) for x in layer_data["router_probs_mean"]],
                    "expert_norms_mean": [float(x) for x in layer_data["expert_norms_mean"]],
                    "sample_count": int(layer_data["sample_count"]),
                    "model_layer_idx": int(layer_data["model_layer_idx"]),
                }
            json_file = save_path / f"expert_analysis_results_sample_{sample_idx}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Sample {sample_idx} results saved to: {json_file}")

    logger.info("Analysis complete!")
    logger.info(f"Results output directory: {save_path}")
    logger.info(f"Generated {len(samples)} JSON files (expert_analysis_results_sample_*.json)")

    return AnalysisResult(output_dir=save_path, model_name=model_name)
