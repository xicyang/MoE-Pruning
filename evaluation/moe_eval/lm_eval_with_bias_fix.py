"""lm-eval wrapper script that automatically loads gate.bias parameters.

Before calling lm-eval, injects bias loading logic through monkey patching.

Usage:
    python -m moe_eval.lm_eval_with_bias_fix --model hf --model_args pretrained=MODEL_PATH --tasks TASK ...
"""

import logging
import os
import sys
from pathlib import Path

# Add the package directory to path to import load_bias_fix
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

# Add the lm_eval directory to path for local imports
lm_eval_dir = Path(__file__).parent / "lm_eval"
sys.path.insert(0, str(lm_eval_dir))

logger = logging.getLogger(__name__)


def _set_use_legacy_cache_default(module):
    """Ensure global use_legacy_cache symbol exists to avoid NameError in patched files."""
    try:
        module.__dict__["use_legacy_cache"] = False
    except Exception:
        pass


def _inject_use_legacy_cache_flag(func):
    """Inject use_legacy_cache global into a function if needed."""
    try:
        if func and hasattr(func, "__globals__"):
            func.__globals__["use_legacy_cache"] = False
    except Exception:
        pass


def _ensure_use_legacy_cache_symbol():
    """Iterate through loaded modules and ensure fallback symbol exists."""
    import sys

    for module in list(sys.modules.values()):
        try:
            if module and hasattr(module, "__name__"):
                _set_use_legacy_cache_default(module)
        except Exception:
            continue


def patch_hflm_create_model():
    """Patch HFLM._create_model method to automatically load gate.bias after model loading."""
    # Import lm_eval module
    from lm_eval.models.huggingface import HFLM

    # Save original method
    original_create_model = HFLM._create_model

    def patched_create_model(self, pretrained, *args, **kwargs):
        """Wrapped _create_model method."""
        # Call original method to load model
        original_create_model(self, pretrained, *args, **kwargs)

        # Load gate.bias for MoE models
        model = self._model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            first_layer = model.model.layers[0]
            has_moe = False

            # Mixtral uses block_sparse_moe
            if hasattr(first_layer, "block_sparse_moe") and hasattr(first_layer.block_sparse_moe, "gate"):
                has_moe = True
            # OLMoE uses mlp
            elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate"):
                has_moe = True

            if has_moe:
                model_path = None
                if pretrained and isinstance(pretrained, str) and os.path.exists(pretrained):
                    model_path = pretrained
                elif hasattr(self, "pretrained") and isinstance(self.pretrained, str) and os.path.exists(self.pretrained):
                    model_path = self.pretrained
                elif hasattr(self, "_pretrained_path") and os.path.exists(self._pretrained_path):
                    model_path = self._pretrained_path

                if model_path:
                    try:
                        if hasattr(first_layer, "block_sparse_moe"):
                            model_type = "Mixtral"
                        elif hasattr(first_layer, "mlp") and hasattr(first_layer.mlp, "gate"):
                            model_type = "OLMoE"
                        else:
                            model_type = "MoE"
                        logger.info(f"Loading gate.bias for {model_type} model: {model_path}")
                        from moe_eval.load_bias_fix import load_gate_bias_from_checkpoint
                        load_gate_bias_from_checkpoint(model, model_path)
                        logger.info("gate.bias loading complete")
                    except Exception as e:
                        logger.warning(f"Error loading gate.bias: {e}")
                else:
                    logger.debug("No valid model path found, skipping gate.bias loading")

    # Replace method
    HFLM._create_model = patched_create_model
    logger.debug("Successfully patched HFLM._create_model method")

    # Also patch __init__ method to save pretrained path
    original_init = HFLM.__init__

    def patched_init(self, pretrained, *args, **kwargs):
        """Wrapped __init__ method, saves pretrained path."""
        if isinstance(pretrained, str):
            self._pretrained_path = pretrained
        original_init(self, pretrained, *args, **kwargs)

    HFLM.__init__ = patched_init
    logger.debug("Successfully patched HFLM.__init__ method")


# Patch HFLM (always needed for gate.bias loading)
patch_hflm_create_model()

# Now import and run lm_eval
if __name__ == "__main__":
    # Set logging (use WARNING level to reduce output unless debugging is needed)
    log_level = os.environ.get("LM_EVAL_BIAS_FIX_LOG_LEVEL", "WARNING")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Import and run lm_eval's main function
    try:
        from lm_eval.__main__ import cli_evaluate
    except ImportError as e:
        print(f"Error: Cannot import lm_eval module: {e}", file=sys.stderr)
        print("Please ensure lm-evaluation-harness is installed", file=sys.stderr)
        sys.exit(1)

    # Run lm_eval
    cli_evaluate()
