# MoE Pruning and Evaluation

Official implementation of **MoE Pathfinder: Trajectory-driven Expert Pruning**, containing two independent modules: pruning and evaluation.

## Project Structure

```
code/
├── pruning/                    # Pruning module
│   ├── README.md              # Pruning usage guide
│   ├── pyproject.toml         # Project configuration
│   └── moe_prune/         # Pruning code
│       ├── __init__.py
│       ├── config.py          # Configuration file
│       ├── cli/               # CLI entry point
│       │   ├── main.py
│       │   └── __init__.py
│       ├── models/            # Model loading
│       │   ├── loader.py
│       │   └── __init__.py
│       ├── analysis/          # Expert analysis
│       │   ├── analyzer.py
│       │   ├── expert_analyzer.py
│       │   └── __init__.py
│       ├── pruning/           # Pruning implementation
│       │   ├── pruner.py
│       │   └── __init__.py
│       ├── datasets/          # Dataset loading
│       │   ├── loaders.py
│       │   └── __init__.py
│       ├── workflows/         # Workflows
│       │   ├── freq_rank_flow.py
│       │   └── __init__.py
│       └── utils/             # Utility modules
│           ├── logging.py
│           └── __init__.py
│
├── evaluation/                 # Independent evaluation module
│   ├── pyproject.toml         # Project configuration
│   ├── moe_eval/          # Evaluation code
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration file
│   │   ├── evaluator.py       # Evaluation entry point
│   │   ├── lm_eval_with_bias_fix.py
│   │   ├── load_bias_fix.py   # gate.bias loading
│   │   ├── cli/               # CLI entry point
│   │   │   ├── main.py
│   │   │   └── __init__.py
│   │   ├── utils/             # Utility modules
│   │   │   ├── logging.py
│   │   │   └── __init__.py
│   │   └── README.md          # Evaluation module guide
│
└── README.md                   # Main documentation
```

## Important Preliminary Step
This evaluation module relies on a modified lm_evalcomponent. Before running the evaluation, you must manually download​ the lm_evalfolder​ from the EleutherAI/lm-evaluation-harnessrepository and place it entirely within the following project path:

```bash
{Your_Project_Root}/evaluation/moe_eval/lm_eval/
```
Please ensure the downloaded lm_evalfolder contains core subdirectories and files such as __init__.py, evaluator.py, models/, etc. This replaces or supplements the corresponding modules in the standard lm-evallibrary to enable correct evaluation for pruned MoE models.

## Supported Models

This toolkit supports MoE models with standard router-based architectures. It has been tested with:
- `Mixtral-8x7B-v0.1` (base model)
- `Mixtral-8x7B-Instruct-v0.1` (instruction-tuned model)

Other MoE models with similar architectures should work with minimal modifications.

## Environment Requirements

### Pruning Environment (moeprune2)

For running analysis, path search, and pruning steps.

```bash
conda create -n moeprune2 python=3.10
conda activate moeprune2
pip install torch transformers accelerate datasets numpy scikit-learn tqdm
```

### Evaluation Environment (eval)

For running model evaluation (requires lm-evaluation-harness).

```bash
conda create -n eval python=3.10
conda activate eval
pip install torch transformers accelerate datasets tqdm
pip install lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git
```

## Quick Start

### 1. Pruning

#### Python API

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
)
```

#### Command Line

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

### 2. Evaluation

#### Using Independent Evaluation Module

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

## Workflow

### Pruning Steps

1. **Analysis**: Use k-means clustering and calibration data to analyze experts
2. **Path Search**: Frequency-based path search to determine which experts to keep
3. **Pruning**: Zero out pruned expert weights and disable router routing
4. **Evaluation**: Evaluate pruned model on benchmark tasks

### gate.bias Loading

Pruned models automatically load `gate.bias` parameters from checkpoints. This is implemented through monkey-patching lm-evaluation-harness to ensure pruned experts are correctly disabled.

## Parameter Description

### Pruning Parameters

> Note: `n_S` parameter is fixed to 1, no need to adjust.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `k` | K-means cluster count | 5 |
| `target_keep` | Number of experts to keep per layer | 128 |
| `path_topk` | Top paths per sample | 10 |
| `path_limit` | Maximum samples for path search | 25 |

### Calibration Datasets

Available datasets: `mmlu`, `arc`, `medqa`, `winogrande`, `hellaswag`, `gsm8k`

### Evaluation Tasks

- `mmlu` - Multi-task language understanding
- `arc_challenge` - ARC challenge questions
- `winogrande` - Winogrande commonsense reasoning
- `hellaswag` - HellaSwag commonsense reasoning
- `gsm8k` - Math word problems
- `medqa_4options` - Medical question answering

## Output Structure

By default, results are saved in the current directory:

```
# Pruning results (default: ./pruneresult/{model_name}/{dataset}/analyze/)
./pruneresult/{model_name}/{dataset}/analyze/    # Analysis results
./prunemodel/{model_name}/{dataset}/             # Pruned models

# Evaluation results (requires manual output directory specification)
{output_dir}/{task}/                              # Evaluation results
```

### Configuring Output Paths

#### Method 1: Using Command Line Arguments (Recommended)

```bash
# Specify output paths for analysis results and pruned models
python -m moe_prune.cli.main \
    --model-path /path/to/your-moe-model \
    --output-base /path/to/results \
    --pruned-model-dir /path/to/pruned_models \
    --calibration-datasets mmlu
```

#### Method 2: Using Environment Variables

```bash
# Set base paths
export MOE_DATA_ROOT=/path/to/data
export MOE_RESULT_BASE=/path/to/results
export MOE_MODEL_BASE=/path/to/pruned_models

# Set dataset paths (optional)
export MOE_DATASET_MMLU=/path/to/mmlu
export MOE_DATASET_ARC=/path/to/arc

# Set model alias paths (optional)
export MOE_MODEL_YOUR_MODEL=/path/to/your-model
```

## Dependencies

### Pruning Dependencies

- torch=2.9.0
- transformers>=4.44.0
- accelerate>=0.34.0
- datasets>=2.19.0
- numpy>=1.24.0
- scikit-learn>=1.4.1
- tqdm>=4.66.0

### Evaluation Dependencies

- lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git
- evaluate>=0.4.1

## License

MIT
