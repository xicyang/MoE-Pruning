"""Fix function: Manually load gate.bias after model loading.

Transformers ignores gate.bias when loading MoE models (original models don't have bias).
Supports both Mixtral and OLMoE models.
"""

import logging
from pathlib import Path

import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def get_moe_block(layer):
    """Get MoE block, compatible with Mixtral and OLMoE."""
    # Mixtral uses block_sparse_moe
    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe, "block_sparse_moe"
    # OLMoE uses mlp
    if hasattr(layer, "mlp"):
        return layer.mlp, "mlp"
    raise ValueError("Cannot find MoE block in layer. Expected 'block_sparse_moe' (Mixtral) or 'mlp' (OLMoE)")


def load_gate_bias_from_checkpoint(model, model_path):
    """
    Manually load gate.bias from checkpoint files and add to model.

    Args:
        model: Loaded model
        model_path: Model path (directory containing safetensors files)
    """
    model_path = Path(model_path)

    # Find all safetensors files
    safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        logger.warning(f"No safetensors files found in {model_path}")
        return

    logger.info(f"Loading gate.bias from {len(safetensors_files)} safetensors files...")

    num_layers = len(model.model.layers)
    loaded_count = 0

    # Detect model type (from first layer)
    first_layer = model.model.layers[0]
    _, moe_block_name = get_moe_block(first_layer)
    logger.info(f"Detected model type: MoE block name = '{moe_block_name}'")

    # Iterate through all safetensors files
    for safetensors_file in safetensors_files:
        try:
            data = safetensors.torch.load_file(str(safetensors_file))

            # Find all gate.bias keys (support Mixtral and OLMoE)
            for key, bias_tensor in data.items():
                # Support two formats:
                # - model.layers.{layer_idx}.block_sparse_moe.gate.bias (Mixtral)
                # - model.layers.{layer_idx}.mlp.gate.bias (OLMoE)
                if f"{moe_block_name}.gate.bias" in key:
                    # Parse layer index
                    # key format: model.layers.{layer_idx}.{moe_block_name}.gate.bias
                    parts = key.split(".")
                    if len(parts) >= 2 and parts[0] == "model" and parts[1] == "layers":
                        try:
                            layer_idx = int(parts[2])
                            if 0 <= layer_idx < num_layers:
                                layer = model.model.layers[layer_idx]
                                moe_block, _ = get_moe_block(layer)

                                # Ensure gate has bias parameter
                                if moe_block.gate.bias is None:
                                    # Create bias parameter using gate.weight's device and dtype
                                    gate_device = moe_block.gate.weight.device
                                    gate_dtype = moe_block.gate.weight.dtype
                                    moe_block.gate.bias = torch.nn.Parameter(
                                        torch.zeros(
                                            moe_block.gate.out_features,
                                            dtype=gate_dtype,
                                            device=gate_device,
                                        )
                                    )

                                # Load bias value, ensure on correct device
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

    # Verify loading results
    verified_count = 0
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        moe_block, _ = get_moe_block(layer)
        if moe_block.gate.bias is not None:
            # Check if there are pruned experts (bias has extreme values)
            bias = moe_block.gate.bias.data
            pruned_experts = (bias < -1e10).sum().item()
            if pruned_experts > 0:
                verified_count += 1
                logger.debug(f"Layer {layer_idx}: {pruned_experts} experts have pruned bias values")

    if verified_count > 0:
        logger.info(f"Verified: {verified_count} layers have pruned experts (bias < -1e10)")
    else:
        logger.warning(
            "No pruned experts found in loaded biases. "
            "This might indicate the model was not pruned or biases were not saved correctly."
        )
