# MoE Model Evaluation

Evaluation module: Tools for evaluating MoE (Mixture of Experts) models with automatic gate.bias loading support.

## Features

- Support for MoE models with standard router-based architectures
- Automatically load gate.bias parameters from checkpoints
- **Self-contained complete lm-evaluation-harness**, no external dependencies required
- Fully compatible with lm-evaluation-harness

## Supported Models

This toolkit has been tested with:
- `Mixtral-8x7B-v0.1` (base model)
- `Mixtral-8x7B-Instruct-v0.1` (instruction-tuned model)
- `OLMoE-1B-7B-0924`

Other MoE models with similar architectures should work with minimal modifications.

## Installation

### Evaluation Environment (eval)

```bash
# Activate conda environment
conda activate eval

# Install package (now includes complete lm-evaluation-harness)
cd evaluation
pip install -e .
```

## Usage

### Command Line

```bash
# Activate evaluation environment
conda activate eval

# Run evaluation
python -m moe_eval.cli.main \
    --model-path /path/to/pruned_model \
    --tasks mmlu arc_challenge winogrande hellaswag \
    --output-dir /path/to/eval_results \
    --cuda-devices 0,1,2,3 \
    --batch-size 8
```

### Python API

```python
from pathlib import Path
from moe_eval.evaluator import run_evaluation

run_evaluation(
    model_path=Path("/path/to/pruned_model"),
    eval_tasks=["mmlu", "arc_challenge"],
    output_dir=Path("/path/to/eval_results"),
    cuda_devices="0,1,2,3",
    batch_size=8,
)
```

## gate.bias Loading

Pruned MoE models store pruning information in `gate.bias` parameters. This module automatically loads these parameters when evaluating models, ensuring that pruned experts are correctly disabled.

The loading is implemented through monkey-patching lm-evaluation-harness, so it works transparently with all evaluation tasks.

## Evaluation Tasks

Available tasks:
- `mmlu` - Multi-task language understanding
- `arc_challenge` - ARC challenge questions
- `winogrande` - Winogrande commonsense reasoning
- `hellaswag` - HellaSwag commonsense reasoning
- `gsm8k` - Math word problems
- `medqa_4options` - Medical question answering

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-path` | Path to the model to evaluate | Required |
| `--tasks` | Evaluation tasks | All common tasks |
| `--output-dir` | Directory to save evaluation results | Required |
| `--cuda-devices` | CUDA devices | 0,1,2,3 |
| `--batch-size` | Batch size for evaluation | 8 |
| `--hf-endpoint` | HuggingFace endpoint | https://hf-mirror.com |
| `--python` | Python executable to use | System default |

## Dependencies

- lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git
- evaluate>=0.4.1
- torch
- transformers
- accelerate

## License

MIT
