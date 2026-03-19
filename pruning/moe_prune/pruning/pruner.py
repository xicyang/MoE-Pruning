"""Pruning: Zero out pruned experts."""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_prune.config import PruningConfig
from moe_prune.models.loader import load_model_bundle
from moe_prune.utils import get_logger

logger = get_logger(__name__)


def get_num_experts(config) -> int:
    if hasattr(config, "num_local_experts") and config.num_local_experts is not None:
        return config.num_local_experts
    if hasattr(config, "num_experts") and config.num_experts is not None:
        return config.num_experts
    raise ValueError("Cannot determine number of experts from model config")


def get_moe_block(layer) -> torch.nn.Module:
    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe
    if hasattr(layer, "mlp"):
        return layer.mlp
    raise ValueError("Cannot find MoE block in layer. Expected 'block_sparse_moe' (Mixtral)")


def detect_moe_layers(model: AutoModelForCausalLM) -> List[int]:
    moe_layers: List[int] = []
    for idx, layer in enumerate(model.model.layers):
        try:
            moe_block = get_moe_block(layer)
        except ValueError:
            continue

        if hasattr(moe_block, "experts"):
            moe_layers.append(idx)
    return moe_layers


@dataclass
class PruningResult:
    pruned_model_dir: Path


def load_pruning_mask(mask_path: Path) -> Dict:
    logger.info(f"Loading pruning mask from {mask_path}")
    with open(mask_path, "r") as f:
        mask_data = json.load(f)

    pruning_masks = mask_data["pruning_masks"]

    total_experts = 0
    pruned_experts = 0

    for layer_key, mask in pruning_masks.items():
        total_experts += len(mask)
        pruned_experts += sum(1 for m in mask if m == 0)

    logger.info(f"Mask statistics:")
    logger.info(f"  Total experts: {total_experts}")
    logger.info(f"  Experts to prune: {pruned_experts}")
    logger.info(f"  Experts to keep: {total_experts - pruned_experts}")
    logger.info(f"  Pruning ratio: {pruned_experts / total_experts * 100:.2f}%")

    return mask_data


def zero_out_expert_weights_and_router(
    model: AutoModelForCausalLM,
    pruning_masks: Dict[str, List[int]],
    router_logit_value: float = -1e20,
    moe_layer_indices: List[int] | None = None,
) -> AutoModelForCausalLM:
    logger.info("Zeroing out pruned experts and disabling router routing...")
    logger.info(f"  Router logit value for pruned experts: {router_logit_value}")

    num_layers = len(model.model.layers)
    target_layers = moe_layer_indices or list(range(num_layers))

    for logical_idx, layer_idx in enumerate(target_layers):
        layer = model.model.layers[layer_idx]
        moe_block = get_moe_block(layer)

        layer_key = f"layer_{logical_idx}"
        if layer_key not in pruning_masks:
            logger.warning(f"No mask found for {layer_key}, skipping")
            continue

        mask = pruning_masks[layer_key]
        experts_to_prune = [i for i, m in enumerate(mask) if m == 0]
        experts_to_keep = [i for i, m in enumerate(mask) if m == 1]

        # Zero out expert weights
        for expert_idx in experts_to_prune:
            expert = moe_block.experts[expert_idx]
            with torch.no_grad():
                if hasattr(expert, "w1"):
                    expert.w1.weight.zero_()
                if hasattr(expert, "w2"):
                    expert.w2.weight.zero_()
                if hasattr(expert, "w3"):
                    expert.w3.weight.zero_()

        # Disable router routing
        with torch.no_grad():
            gate_bias: torch.nn.Parameter | None = getattr(moe_block.gate, "bias", None)
            if gate_bias is None:
                bias_param = torch.nn.Parameter(
                    torch.zeros(
                        moe_block.gate.weight.shape[0],
                        dtype=moe_block.gate.weight.dtype,
                        device=moe_block.gate.weight.device,
                    )
                )
                moe_block.gate.bias = bias_param
                gate_bias = bias_param

            for expert_idx in experts_to_prune:
                moe_block.gate.weight[expert_idx, :].zero_()
                if gate_bias is not None:
                    gate_bias.data[expert_idx] = float(router_logit_value)

    logger.info("All pruned experts zeroed and router routing disabled")
    return model


def verify_pruning(
    model: AutoModelForCausalLM,
    pruning_masks: Dict[str, List[int]],
    router_logit_value: float = -1e20,
    test_input_size: int = 128,
    moe_layer_indices: List[int] | None = None,
) -> bool:
    logger.info("Verifying pruning...")

    num_layers = len(model.model.layers)
    all_correct = True

    target_layers = moe_layer_indices or list(range(num_layers))
    first_layer = model.model.layers[target_layers[0]]
    first_moe_block = get_moe_block(first_layer)
    hidden_size = first_moe_block.gate.weight.shape[1]

    for logical_idx, layer_idx in enumerate(target_layers):
        layer = model.model.layers[layer_idx]
        moe_block = get_moe_block(layer)

        layer_key = f"layer_{logical_idx}"
        if layer_key not in pruning_masks:
            continue

        mask = pruning_masks[layer_key]
        experts_to_prune = [i for i, m in enumerate(mask) if m == 0]

        for expert_idx in experts_to_prune:
            expert = moe_block.experts[expert_idx]

            w1_is_zero = torch.all(expert.w1.weight == 0).item()
            w2_is_zero = torch.all(expert.w2.weight == 0).item()
            w3_is_zero = torch.all(expert.w3.weight == 0).item()
            expert_weights_zero = w1_is_zero and w2_is_zero and w3_is_zero

            gate_weight_is_zero = torch.all(moe_block.gate.weight[expert_idx, :] == 0).item()

            bias_matches = True
            gate_bias: torch.nn.Parameter | None = getattr(moe_block.gate, "bias", None)
            if gate_bias is not None:
                expected_bias = torch.tensor(
                    router_logit_value,
                    device=gate_bias.device,
                    dtype=gate_bias.dtype,
                )
                bias_matches = torch.isclose(
                    gate_bias[expert_idx],
                    expected_bias,
                    atol=abs(router_logit_value) * 0.01,
                ).item()

            test_input = torch.randn(
                test_input_size,
                hidden_size,
                dtype=moe_block.gate.weight.dtype,
                device=moe_block.gate.weight.device,
            )

            router_logits = torch.matmul(test_input, moe_block.gate.weight.t())
            gate_bias = getattr(moe_block.gate, "bias", None)
            if gate_bias is not None:
                router_logits = router_logits + gate_bias
            pruned_logits = router_logits[:, expert_idx]
            all_logits_negative = torch.all(pruned_logits < -1e10).item()

            if not (expert_weights_zero and gate_weight_is_zero and bias_matches and all_logits_negative):
                logger.error(f"Layer {layer_idx}, Expert {expert_idx}: NOT properly pruned!")
                all_correct = False

    if all_correct:
        logger.info("Verification passed: all pruned experts properly zeroed")
    else:
        logger.error("Verification failed: some experts not properly pruned")

    return all_correct


def calculate_model_size(model: AutoModelForCausalLM) -> Dict:
    total_params = 0
    non_zero_params = 0

    for p in model.parameters():
        if p.device.type == "meta":
            continue

        param_count = p.numel()
        total_params += param_count

        try:
            non_zero_count = (p != 0).sum().item()
            non_zero_params += non_zero_count
        except RuntimeError as e:
            if "meta" in str(e).lower():
                logger.warning(f"Skipping parameter on meta device: {e}")
                continue
            raise

    logger.info(f"Model size:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Non-zero parameters: {non_zero_params:,}")
    logger.info(f"  Zero parameters: {total_params - non_zero_params:,}")
    if total_params > 0:
        logger.info(f"  Effective sparsity: {(total_params - non_zero_params) / total_params * 100:.2f}%")
    else:
        logger.warning("  No parameters found (all may be on meta device)")

    return {
        "total_params": total_params,
        "non_zero_params": non_zero_params,
        "zero_params": total_params - non_zero_params,
        "sparsity": (total_params - non_zero_params) / total_params if total_params > 0 else 0.0,
    }


def save_pruning_info(save_path: Path, mask_data: Dict, model_size_info: Dict, router_logit_value: float):
    info = {
        "pruning_method": "zero_out",
        "description": "Pruned experts zeroed, router routing disabled.",
        "router_logit_value": router_logit_value,
        "model_size": model_size_info,
        "pruning_masks": mask_data["pruning_masks"],
        "timestamp": datetime.now().isoformat(),
    }

    info_path = save_path / "pruning_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Pruning info saved to {info_path}")


def apply_pruning_mask(
    model_path: Path,
    mask_path: Path,
    config: PruningConfig,
) -> PruningResult:
    logger.info("=" * 80)
    logger.info("Pruning: Zero-out Method")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Mask path: {mask_path}")
    logger.info(f"Output base: {config.output_base}")
    logger.info(f"Router logit value: {config.router_logit_value}")
    logger.info("=" * 80)

    mask_data = load_pruning_mask(mask_path)

    model_name = model_path.name
    mask_name = mask_path.stem
    save_path = config.output_base / f"{model_name}_{mask_name}_nS{config.n_s}"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model...")
    bundle = load_model_bundle(model_path, dtype="bfloat16", device_map="auto")
    model = bundle.model
    tokenizer = bundle.tokenizer

    num_experts = get_num_experts(model.config)
    logger.info(f"Model loaded: {len(model.model.layers)} layers, {num_experts} experts per layer")

    original_size = calculate_model_size(model)

    # Get MoE layer indices
    mask_moe_layers = mask_data.get("moe_layer_indices")
    detected_moe_layers = detect_moe_layers(model)
    if mask_moe_layers:
        moe_layer_indices = mask_moe_layers
        if len(moe_layer_indices) != len(mask_data["pruning_masks"]):
            logger.warning(
                "Mask moe_layer_indices count (%d) differs from pruning_masks layers (%d), aligning to mask layer count.",
                len(moe_layer_indices),
                len(mask_data["pruning_masks"]),
            )
    else:
        moe_layer_indices = detected_moe_layers
        logger.info("Mask does not provide moe_layer_indices, using auto-detected: %s", moe_layer_indices)
        if len(moe_layer_indices) != len(mask_data["pruning_masks"]):
            logger.warning(
                "Auto-detected MoE layer count (%d) differs from pruning_masks layers (%d), truncating to mask layer count.",
                len(moe_layer_indices),
                len(mask_data["pruning_masks"]),
            )
            moe_layer_indices = moe_layer_indices[: len(mask_data["pruning_masks"])]

    model = zero_out_expert_weights_and_router(
        model,
        mask_data["pruning_masks"],
        router_logit_value=config.router_logit_value,
        moe_layer_indices=moe_layer_indices,
    )

    if config.verify:
        verification_passed = verify_pruning(
            model,
            mask_data["pruning_masks"],
            router_logit_value=config.router_logit_value,
            moe_layer_indices=moe_layer_indices,
        )
        if not verification_passed:
            logger.error("Verification failed! Model will still be saved.")

    pruned_size = calculate_model_size(model)

    logger.info("Saving pruned model...")
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    save_pruning_info(save_path, mask_data, pruned_size, config.router_logit_value)

    logger.info("Pruning complete!")
    logger.info(f"Pruned model saved to: {save_path}")

    return PruningResult(pruned_model_dir=save_path)