# MoE Evaluation Module

用于评估剪枝后的MoE模型。

## 使用方法

```bash
python -m moe_eval.cli.main \
    --model-path /path/to/pruned_model \
    --tasks mmlu arc_challenge winogrande hellaswag \
    --output-dir /path/to/eval_results \
    --cuda-devices 0,1,2,3 \
    --batch-size 8
```

## 任务列表

- `mmlu` - 多任务语言理解
- `arc_challenge` - ARC挑战集
- `winogrande` - Winogrande常识推理
- `hellaswag` - HellaSwag常识推理
- `gsm8k` - 数学应用题
- `medqa_4options` - 医疗问答