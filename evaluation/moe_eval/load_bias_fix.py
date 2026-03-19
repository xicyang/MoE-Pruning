"""Load gate.bias from checkpoint for pruned MoE models."""

import logging
from pathlib import Path

import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def get_moe_block(layer):
    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe, "block_sparse_moe"
    if hasattr(layer, "mlp"):
        return layer.mlp, "mlp"
    raise ValueError("Cannot find MoE block in layer")


def load_gate_bias_from_checkpoint(model, model_path):
    model_path = Path(model_path)

    safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        logger.warning(f"No safetensors files found in {model_path}")
        return

    logger.info(f"Loading gate.bias from {len(safetensors_files)} safetensors files...")

    num_layers = len(model.model.layers)
    loaded_count = 0

    first_layer = model.model.layers[0]
    _, moe_block_name = get_moe_block(first_layer)
    logger.info(f"Detected model type: MoE block name = '{moe_block_name}'")

    for safetensors_file in safetensors_files:
        try:
            data = safetensors.torch.load_file(str(safetensors_file))

            for key, bias_tensor in data.items():
                if f"{moe_block_name}.gate.bias" in key:
                    parts = key.split(".")
                    if len(parts) >= 2 and parts[0] == "model" and parts[1] == "layers":
                        try:
                            layer_idx = int(parts[2])
                            if 0 <= layer_idx < num_layers:
                                layer = model.model.layers[layer_idx]
                                moe_block, _ = get_moe_block(layer)

                                if moe_block.gate.bias is None:
                                    gate_device = moe_block.gate.weight.device
                                    gate_dtype = moe_block.gate.weight.dtype
                                    moe_block.gate.bias = torch.nn.Parameter(
                                        torch.zeros(
                                            moe_block.gate.out_features,
                                            dtype=gate_dtype,
                                            device=gate_device,
                                        )
                                    )

                                gate_device = moe_block.gate.weight.device
                                if bias_tensor.device != gate_device:
                                    bias_tensor = bias_tensor.to(gate_device)
                                moe_block.gate.bias.data.copy_(bias_tensor)
                                loaded_count += 1

                                logger.debug(f"Loaded bias for layer {layer_idx}: {bias_tensor[:3].tolist()}...")
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse layer index from key {key}: {e}")
                            continue
        except Exception as e:
            logger.warning(f"Error loading {safetensors_file}: {e}")
            continue

    logger.info(f"Loaded gate.bias for {loaded_count} layers")

    verified_count = 0
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        moe_block, _ = get_moe_block(layer)
        if moe_block.gate.bias is not None:
            bias = moe_block.gate.bias.data
            pruned_experts = (bias < -1e10).sum().item()
            if pruned_experts > 0:
                verified_count += 1
                logger.debug(f"Layer {layer_idx}: {pruned_experts} experts have pruned bias values")

    if verified_count > 0:
        logger.info(f"Verified: {verified_count} layers have pruned experts")
    else:
        logger.warning(
            "No pruned experts found in loaded biases. "
            "This might indicate the model was not pruned or biases were not saved correctly."
        )