"""Dataset loading utilities."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset, load_from_disk

from moe_prune.config import DATASET_PATHS

logger = logging.getLogger(__name__)


def load_multi_domain_datasets(
    dataset_to_load: str,
    mmlu_path: str | Path | None = None,
    arc_path: str | Path | None = None,
    medqa_path: str | Path | None = None,
    winogrande_path: str | Path | None = None,
    hellaswag_path: str | Path | None = None,
    gsm8k_path: str | Path | None = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Load data from specified dataset and format uniformly.

    Args:
        dataset_to_load: Dataset name: 'mmlu', 'arc', 'medqa', 'winogrande', 'hellaswag', 'gsm8k'
        mmlu_path: MMLU dataset path
        arc_path: ARC-Challenge dataset path
        medqa_path: MedQA-USMLE dataset path
        winogrande_path: Winogrande dataset path
        hellaswag_path: HellaSwag dataset path
        gsm8k_path: GSM8K dataset path
        split: Dataset split (train/validation/test)

    Returns:
        Formatted data list, each element contains 'question' and 'answer' fields
    """
    # Use default paths from config if not provided
    if mmlu_path is None:
        mmlu_path = str(DATASET_PATHS["mmlu"])
    if arc_path is None:
        arc_path = str(DATASET_PATHS["arc"])
    if medqa_path is None:
        medqa_path = str(DATASET_PATHS["medqa"])
    if winogrande_path is None:
        winogrande_path = str(DATASET_PATHS["winogrande"])
    if hellaswag_path is None:
        hellaswag_path = str(DATASET_PATHS["hellaswag"])
    if gsm8k_path is None:
        gsm8k_path = str(DATASET_PATHS["gsm8k"])

    all_data = []

    valid_datasets = {"mmlu", "arc", "medqa", "winogrande", "hellaswag", "gsm8k"}
    if dataset_to_load not in valid_datasets:
        raise ValueError(f"Invalid dataset name: {dataset_to_load}. Valid: {valid_datasets}")

    # Load MMLU
    if dataset_to_load == "mmlu":
        try:
            mmlu_split_name = "auxiliary_train" if split == "train" else split
            mmlu_split = None

            try:
                mmlu_ds = load_from_disk(mmlu_path)
                available_splits = list(mmlu_ds.keys())
                if mmlu_split_name in available_splits:
                    mmlu_split = mmlu_ds[mmlu_split_name]
                elif split in available_splits:
                    mmlu_split = mmlu_ds[split]
                    mmlu_split_name = split
            except (KeyError, ValueError, OSError, FileNotFoundError):
                mmlu_split = None

            if mmlu_split is None:
                parquet_file = os.path.join(mmlu_path, f"{mmlu_split_name}-00000-of-00001.parquet")
                if not os.path.exists(parquet_file) and split == "train":
                    for name in ["auxiliary_train", "train"]:
                        test_file = os.path.join(mmlu_path, f"{name}-00000-of-00001.parquet")
                        if os.path.exists(test_file):
                            parquet_file = test_file
                            mmlu_split_name = name
                            break
                    else:
                        raise FileNotFoundError(f"MMLU training set not found")
                elif not os.path.exists(parquet_file):
                    raise FileNotFoundError(f"MMLU parquet file not found: {parquet_file}")

                mmlu_ds = load_dataset("parquet", data_files=parquet_file)
                mmlu_split = mmlu_ds["train"]

            for item in mmlu_split:
                question = item["question"]
                choices = item["choices"]
                answer_idx = item["answer"]
                answer = choices[answer_idx] if isinstance(answer_idx, int) else choices[0]
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                formatted_question = f"{question}\n\n{choices_text}"
                all_data.append({"question": formatted_question, "answer": answer, "source": "mmlu"})
        except Exception as e:
            logger.error(f"Failed to load MMLU: {e}", exc_info=True)

    # Load ARC
    if dataset_to_load == "arc":
        try:
            parquet_file = os.path.join(arc_path, f"{split}-00000-of-00001.parquet")
            if not os.path.exists(parquet_file):
                raise FileNotFoundError(f"ARC parquet file not found: {parquet_file}")
            arc_ds = load_dataset("parquet", data_files=parquet_file)
            arc_split = arc_ds["train"]

            for item in arc_split:
                question = item["question"]
                choices = item["choices"]
                answer_key = item["answerKey"]

                if isinstance(choices, dict):
                    choice_texts = choices.get("text", [])
                    choice_labels = choices.get("label", [])
                    choices_text = "\n".join([f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)])
                    answer_idx = choice_labels.index(answer_key) if answer_key in choice_labels else 0
                    answer = choice_texts[answer_idx]
                else:
                    choice_texts = choices if isinstance(choices, list) else []
                    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choice_texts)])
                    answer = choice_texts[0] if choice_texts else ""

                formatted_question = f"{question}\n\n{choices_text}"
                all_data.append({"question": formatted_question, "answer": answer, "source": "arc"})
        except Exception as e:
            logger.error(f"Failed to load ARC: {e}", exc_info=True)

    # Load MedQA
    if dataset_to_load == "medqa":
        try:
            jsonl_file = os.path.join(medqa_path, f"{split}.jsonl")
            if split == "validation":
                jsonl_file = os.path.join(medqa_path, "dev.jsonl")

            if not os.path.exists(jsonl_file):
                raise FileNotFoundError(f"MedQA JSON file not found: {jsonl_file}")

            medqa_data = []
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        medqa_data.append(json.loads(line))

            for item in medqa_data:
                question = item["sent1"]
                endings = [item.get(f"ending{i}", "") for i in range(4)]
                label = item.get("label", 0)
                choices_text = "\n".join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(endings) if ending])
                formatted_question = f"{question}\n\n{choices_text}"
                answer = endings[label] if label < len(endings) else endings[0]
                all_data.append({"question": formatted_question, "answer": answer, "source": "medqa"})
        except Exception as e:
            logger.error(f"Failed to load MedQA: {e}", exc_info=True)

    # Load Winogrande
    if dataset_to_load == "winogrande":
        try:
            parquet_file = os.path.join(winogrande_path, f"{split}-00000-of-00001.parquet")
            if not os.path.exists(parquet_file):
                raise FileNotFoundError(f"Winogrande parquet file not found: {parquet_file}")
            winogrande_ds = load_dataset("parquet", data_files=parquet_file)
            winogrande_split = winogrande_ds["train"]

            for item in winogrande_split:
                sentence = item["sentence"]
                option1 = item["option1"]
                option2 = item["option2"]
                answer = item["answer"]
                answer_idx = int(answer) - 1
                options = [option1, option2]
                correct_option = options[answer_idx] if 0 <= answer_idx < len(options) else options[0]
                choices_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                formatted_question = f"{sentence}\n\n{choices_text}"
                all_data.append({"question": formatted_question, "answer": correct_option, "source": "winogrande"})
        except Exception as e:
            logger.error(f"Failed to load Winogrande: {e}", exc_info=True)

    # Load HellaSwag
    if dataset_to_load == "hellaswag":
        try:
            parquet_file = os.path.join(hellaswag_path, f"{split}-00000-of-00001.parquet")
            if not os.path.exists(parquet_file):
                raise FileNotFoundError(f"HellaSwag parquet file not found: {parquet_file}")
            hellaswag_ds = load_dataset("parquet", data_files=parquet_file)
            hellaswag_split = hellaswag_ds["train"]

            for item in hellaswag_split:
                ctx = item["ctx"]
                endings = item["endings"]
                label = item["label"]
                label_idx = int(label) if isinstance(label, str) else label
                correct_ending = endings[label_idx] if 0 <= label_idx < len(endings) else endings[0]
                choices_text = "\n".join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(endings)])
                formatted_question = f"{ctx}\n\n{choices_text}"
                all_data.append({"question": formatted_question, "answer": correct_ending, "source": "hellaswag"})
        except Exception as e:
            logger.error(f"Failed to load HellaSwag: {e}", exc_info=True)

    # Load GSM8K
    if dataset_to_load == "gsm8k":
        try:
            parquet_file = os.path.join(gsm8k_path, f"{split}-00000-of-00001.parquet")
            if not os.path.exists(parquet_file):
                raise FileNotFoundError(f"GSM8K parquet file not found: {parquet_file}")
            gsm8k_ds = load_dataset("parquet", data_files=parquet_file)
            gsm8k_split = gsm8k_ds["train"]

            for item in gsm8k_split:
                question = item["question"]
                answer = item["answer"]
                if "####" in answer:
                    final_answer = answer.split("####")[-1].strip()
                else:
                    final_answer = answer.strip()
                all_data.append({"question": question, "answer": final_answer, "source": "gsm8k"})
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}", exc_info=True)

    if len(all_data) == 0:
        raise RuntimeError(f"Failed to load any data from {dataset_to_load}")

    logger.info(f"Successfully loaded {len(all_data)} samples from {dataset_to_load}")
    return all_data


def format_dataset_texts(
    items: List[Dict[str, Any]], dataset_type: str = "medqa", question_only: bool = False
) -> List[str]:
    """Format dataset items to question+answer or question-only text."""
    formatted: List[str] = []
    for item in items:
        question = item.get("question", "")
        if question_only:
            formatted.append(question)
        else:
            answer = item.get("answer", "")
            formatted.append(f"Question: {question}\nAnswer: {answer}")
    return formatted
