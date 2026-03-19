# MoE Pruning Module

MoE专家剪枝模块，基于频率排序的路径搜索方法。

## 使用方法

### Python API

```python
from pathlib import Path
from moe_prune import run_freq_rank_workflow

run_freq_rank_workflow(
    model_path=Path("/path/to/Mixtral-8x7B"),
    calibration_datasets=["mmlu", "arc"],
    cuda_devices="0,1,2,3",
    analysis_k=5,
    target_keep=128,
    path_topk=10,
    path_limit=25,
)
```

### 命令行

```bash
python -m moe_prune.cli.main \
    --model-path /path/to/Mixtral-8x7B \
    --calibration-datasets mmlu \
    --cuda-devices 0,1,2,3 \
    --analysis-k 5 \
    --target-keep 128 \
    --path-topk 10 \
    --output-base ./results \
    --pruned-model-dir ./pruned_models
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--analysis-k` | K-means聚类数 | 5 |
| `--target-keep` | 每层保留的专家数 | 128 |
| `--path-topk` | 每个样本的top-k路径 | 10 |
| `--path-limit` | 路径搜索的最大样本数 | 25 |

## 输出

- 分析结果: `./pruneresult/{model_name}/{dataset}/analyze/`
- 剪枝模型: `./prunemodel/{model_name}/{dataset}/`