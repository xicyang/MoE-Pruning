# MoE Pathfinder: Trajectory-driven Expert Pruning

[English](#english-version) | [中文](#中文版)

---

<a id="中文版"></a>
# 中文版

**MoE Pathfinder: Trajectory-driven Expert Pruning** 的官方实现，包含两个独立的模块：剪枝（pruning）和评估（evaluation）。

## 项目结构

```text
code/
├── pruning/                    # 剪枝模块
│   ├── README.md              # 剪枝使用指南
│   ├── pyproject.toml         # 项目配置
│   └── moe_prune/             # 剪枝代码
│       ├── __init__.py
│       ├── config.py          # 配置文件
│       ├── cli/               # 命令行入口
│       │   ├── main.py
│       │   └── __init__.py
│       ├── models/            # 模型加载
│       │   ├── loader.py
│       │   └── __init__.py
│       ├── analysis/          # 专家分析
│       │   ├── analyzer.py
│       │   ├── expert_analyzer.py
│       │   └── __init__.py
│       ├── pruning/           # 剪枝实现
│       │   ├── pruner.py
│       │   └── __init__.py
│       ├── datasets/          # 数据集加载
│       │   ├── loaders.py
│       │   └── __init__.py
│       ├── workflows/         # 工作流
│       │   ├── freq_rank_flow.py
│       │   └── __init__.py
│       └── utils/             # 实用工具模块
│           ├── logging.py
│           └── __init__.py
│
├── evaluation/                 # 独立的评估模块
│   ├── pyproject.toml         # 项目配置
│   ├── moe_eval/              # 评估代码
│   │   ├── __init__.py
│   │   ├── config.py          # 配置文件
│   │   ├── evaluator.py       # 评估入口
│   │   ├── lm_eval_with_bias_fix.py
│   │   ├── load_bias_fix.py   # gate.bias 加载
│   │   ├── cli/               # 命令行入口
│   │   │   ├── main.py
│   │   │   └── __init__.py
│   │   ├── utils/             # 实用工具模块
│   │   │   ├── logging.py
│   │   │   └── __init__.py
│   │   └── README.md          # 评估模块指南
│
└── README.md                   # 主文档
```

## 重要前置步骤
此评估模块依赖于经过修改的 `lm_eval` 组件。在运行评估之前，您必须**手动下载** `EleutherAI/lm-evaluation-harness` 仓库中的 `lm_eval` 文件夹，并将其完整放置在以下项目路径中：

```bash
{你的项目根目录}/evaluation/moe_eval/lm_eval/
```
请确保下载的 `lm_eval` 文件夹包含核心的子目录和文件，例如 `__init__.py`、`evaluator.py`、`models/` 等。此操作将替换或补充标准 `lm-eval` 库中的相应模块，以确保对剪枝后的 MoE 模型进行正确的评估。

## 支持的模型

本工具包支持采用标准基于路由（router-based）架构的 MoE 模型。已在以下模型中测试通过：
- `Mixtral-8x7B-v0.1` (基础模型)
- `Mixtral-8x7B-Instruct-v0.1` (指令微调模型)

其他具有类似架构的 MoE 模型只需极少的修改即可正常运行。

## 环境要求

### 剪枝环境 (moeprune2)

用于运行分析、路径搜索和剪枝步骤。

```bash
conda create -n moeprune2 python=3.10
conda activate moeprune2
pip install torch transformers accelerate datasets numpy scikit-learn tqdm
```

### 评估环境 (eval)

用于运行模型评估（需要 lm-evaluation-harness）。

```bash
conda create -n eval python=3.10
conda activate eval
pip install torch transformers accelerate datasets tqdm
pip install lm-eval @ git+[https://github.com/EleutherAI/lm-evaluation-harness.git](https://github.com/EleutherAI/lm-evaluation-harness.git)
```

## 快速开始

### 1. 剪枝 (Pruning)

#### Python API

```python
from pathlib import Path
from moe_prune import run_freq_rank_workflow

# 运行完整的剪枝工作流
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

#### 命令行方式

```bash
# 激活剪枝环境
conda activate moeprune2

# 运行剪枝
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

### 2. 评估 (Evaluation)

#### 使用独立评估模块

```bash
# 激活评估环境
conda activate eval

# 运行评估
python -m moe_eval.cli.main \
    --model-path /path/to/pruned_model \
    --tasks mmlu arc_challenge winogrande hellaswag \
    --output-dir /path/to/eval_results \
    --cuda-devices 0,1,2,3 \
    --batch-size 8
```

## 工作流程

### 剪枝步骤

1. **分析 (Analysis)**: 使用 k-means 聚类和校准数据对专家进行分析。
2. **路径搜索 (Path Search)**: 基于频率的路径搜索，以决定要保留哪些专家。
3. **剪枝 (Pruning)**: 将被剪除的专家权重置零，并禁用路由器的对应路由。
4. **评估 (Evaluation)**: 在基准任务上评估剪枝后的模型。

### gate.bias 加载

剪枝后的模型会自动从检查点加载 `gate.bias` 参数。这是通过对 `lm-evaluation-harness` 打猴子补丁（monkey-patching）来实现的，以确保被剪枝的专家被正确禁用。

## 参数说明

### 剪枝参数

> 注意：`n_S` 参数固定为 1，无需调整。

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `k` | K-means 聚类数 | 5 |
| `target_keep` | 每层目标保留的专家数 | 128 |
| `path_topk` | 每个样本的前 Top-K 路径 | 10 |
| `path_limit` | 路径搜索的最大样本数 | 25 |

### 校准数据集

可用数据集：`mmlu`, `arc`, `medqa`, `winogrande`, `hellaswag`, `gsm8k`

### 评估任务

- `mmlu` - 多任务语言理解
- `arc_challenge` - ARC 挑战集
- `winogrande` - Winogrande 常识推理
- `hellaswag` - HellaSwag 常识推理
- `gsm8k` - 数学应用题
- `medqa_4options` - 医疗问答

## 输出结构

默认情况下，结果保存在当前目录：

```text
# 剪枝结果 (默认: ./pruneresult/{model_name}/{dataset}/analyze/)
./pruneresult/{model_name}/{dataset}/analyze/    # 分析结果
./prunemodel/{model_name}/{dataset}/             # 剪枝后的模型

# 评估结果 (需要手动指定输出目录)
{output_dir}/{task}/                              # 评估结果
```

### 配置输出路径

#### 方法 1: 使用命令行参数 (推荐)

```bash
# 指定分析结果和剪枝后模型的输出路径
python -m moe_prune.cli.main \
    --model-path /path/to/your-moe-model \
    --output-base /path/to/results \
    --pruned-model-dir /path/to/pruned_models \
    --calibration-datasets mmlu
```

#### 方法 2: 使用环境变量

```bash
# 设置基础路径
export MOE_DATA_ROOT=/path/to/data
export MOE_RESULT_BASE=/path/to/results
export MOE_MODEL_BASE=/path/to/pruned_models

# 设置数据集路径 (可选)
export MOE_DATASET_MMLU=/path/to/mmlu
export MOE_DATASET_ARC=/path/to/arc

# 设置模型别名路径 (可选)
export MOE_MODEL_YOUR_MODEL=/path/to/your-model
```

## 依赖项

### 剪枝依赖

- torch=2.9.0
- transformers>=4.44.0
- accelerate>=0.34.0
- datasets>=2.19.0
- numpy>=1.24.0
- scikit-learn>=1.4.1
- tqdm>=4.66.0

### 评估依赖

- lm-eval @ git+https://github.com/EleutherAI/lm-evaluation-harness.git
- evaluate>=0.4.1

## 开源协议

MIT

---

<a id="english-version"></a>
# English Version

Official implementation of **MoE Pathfinder: Trajectory-driven Expert Pruning**, containing two independent modules: pruning and evaluation.

## Project Structure

```text
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
This evaluation module relies on a modified `lm_eval` component. Before running the evaluation, you must manually download the `lm_eval` folder from the `EleutherAI/lm-evaluation-harness` repository and place it entirely within the following project path:

```bash
{Your_Project_Root}/evaluation/moe_eval/lm_eval/
```
Please ensure the downloaded `lm_eval` folder contains core subdirectories and files such as `__init__.py`, `evaluator.py`, `models/`, etc. This replaces or supplements the corresponding modules in the standard `lm-eval` library to enable correct evaluation for pruned MoE models.

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
pip install lm-eval @ git+[https://github.com/EleutherAI/lm-evaluation-harness.git](https://github.com/EleutherAI/lm-evaluation-harness.git)
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

```text
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
