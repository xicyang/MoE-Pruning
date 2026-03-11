# MoE Pruning Pipeline

Pruning module: Expert pruning for MoE (Mixture of Experts) models.

**Note: This module contains only pruning functionality, not evaluation. For evaluation, please use the independent evaluation module.**

## Supported Models

This toolkit supports MoE models with standard router-based architectures. It has been tested with:
- `Mixtral-8x7B-v0.1` (base model)
- `Mixtral-8x7B-Instruct-v0.1` (instruction-tuned model)

Other MoE models with similar architectures should work with minimal modifications.

## Installation

### Pruning Environment (moeprune2)

```bash
# Activate conda environment
conda activate moeprune2

# Install package
cd pruning
pip install -e .
```

## Usage

### Python API

```python
from pathlib import Path
from moe_prune import run_freq_rank_workflow

# Run complete pruning workflow
run_freq_rank_workflow(
    model_path=Path("/path/to/your-moe-model"),
    calibration_datasets=["mmlu", "arc"],
    cuda_devices="0,1,2,3",
    analysis_k=5,
    target_keep=128,
    path_topk=10,
    path_limit=25,
    output_base=Path("/path/to/results"),
    pruned_model_dir=Path("/path/to/pruned_models"),
)
```

### Command Line

```bash
# Activate pruning environment
conda activate moeprune2

# Run pruning
python -m moe_prune.cli.main \
    --model-path /path/to/your-moe-model \
    --calibration-datasets mmlu \
    --cuda-devices 0,1,2,3 \
    --analysis-k 5 \
    --target-keep 128 \
    --path-topk 10 \
    --output-base /path/to/results \
    --pruned-model-dir /path/to/pruned_models
```

## Workflow

The pruning workflow consists of three steps:

1. **Analysis**: Use k-means clustering and calibration data to analyze experts
2. **Path Search**: Frequency-based path search to determine which experts to keep
3. **Pruning**: Zero out pruned expert weights and disable router routing

## Parameters

### Pruning Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--analysis-k` | K-means cluster count | 5 |
| `--target-keep` | Number of experts to keep per layer | 128 |
| `--path-topk` | Top paths per sample | 10 |
| `--path-limit` | Maximum samples for path search | 25 |
| `--output-base` | Base directory for analysis results | `./pruneresult` |
| `--pruned-model-dir` | Directory to save pruned models | `./prunemodel/{model_name}/{dataset}` |

> Note: `n_S` parameter is fixed to 1, no need to adjust.

### Calibration Datasets

Available datasets: `mmlu`, `arc`, `medqa`, `winogrande`, `hellaswag`, `gsm8k`

## Output Structure

By default, results are saved in the current directory:

- Analysis results: `./pruneresult/{model_name}/{dataset}/analyze/`
- Pruned models: `./prunemodel/{model_name}/{dataset}/`

### Configuring Output Paths

#### Method 1: Using Command Line Arguments (Recommended)

```bash
python -m moe_prune.cli.main \
    --model-path /path/to/your-moe-model \
    --output-base /path/to/results \
    --pruned-model-dir /path/to/pruned_models \
    --calibration-datasets mmlu
```

#### Method 2: Using Environment Variables

```bash
export MOE_RESULT_BASE=/path/to/results
export MOE_MODEL_BASE=/path/to/pruned_models
```

## Dependencies

- transformers>=4.44.0
- accelerate>=0.34.0
- datasets>=2.19.0
- numpy>=1.24.0
- scikit-learn>=1.4.1
- tqdm>=4.66.0
