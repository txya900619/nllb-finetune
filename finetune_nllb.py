#!/usr/bin/env python
"""
Basic script to finetune NLLB (M2M100 architecture) with optional new language tag support.

Features:
- Load a pretrained NLLB model and tokenizer from Hugging Face
- Add a new language code token (e.g., yue_Hant) if it's not present
- Prepare custom datasets (CSV/JSON/JSONL/TSV) with configurable source/target columns
- Finetune using Hugging Face Trainer (Seq2Seq)
- Optional LoRA via PEFT for parameter-efficient finetuning

Example:
  python finetune_nllb.py \
    --model_name facebook/nllb-200-distilled-600M \
    --train_file data/train.jsonl \
    --validation_file data/valid.jsonl \
    --source_column src_text \
    --target_column tgt_text \
    --src_lang eng_Latn \
    --tgt_lang yue_Hant \
    --add_missing_lang_codes \
    --output_dir outputs/nllb-yue \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --use_lora

Data format:
- train/validation files can be CSV/TSV/JSON/JSONL with columns for source and target text.

Notes:
- NLLB uses language codes like "eng_Latn" and special tokens internally like "__eng_Latn__".
- When adding a new language code, we add corresponding special token and resize model embeddings.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# Optional import of LANG_GROUP_MAP from prepare_data.py for bulk token registration
try:
    from prepare_data import LANG_GROUP_MAP as PREPARE_LANG_GROUP_MAP
except Exception:
    PREPARE_LANG_GROUP_MAP = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune NLLB with optional new language tag support"
    )

    # Data
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training file (csv/tsv/json/jsonl)",
    )
    parser.add_argument(
        "--to_zh_validation_file",
        type=str,
        required=False,
        default=None,
        help="Path to validation file",
    )
    parser.add_argument(
        "--to_formosan_validation_file",
        type=str,
        required=False,
        default=None,
        help="Path to validation file",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="src_text",
        help="Name of source text column",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="tgt_text",
        help="Name of target text column",
    )

    # Model/tokenizer
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="HF model id or path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="HF tokenizer id or path (defaults to model)",
    )
    parser.add_argument(
        "--prepend_src_lang_token",
        action="store_true",
        help="Prepend language token to source text. Uses per-example src_lang column.",
    )
    parser.add_argument(
        "--prepend_tgt_lang_token",
        action="store_true",
        help="Prepend language token to target text (labels). Uses per-example tgt_lang column.",
    )
    # We always add FAIRSEQ_LANGUAGE_CODES + LANG_GROUP_MAP tokens at load time

    # Training hyperparams
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/finetune-nllb",
        help="Where to save model",
    )
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Logging/saving
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument(
        "--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"]
    )
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--report_to", type=str, default="none", help="wandb/tensorboard/none"
    )

    # LoRA
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient finetuning",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def detect_file_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return "csv"
    if ext in [".tsv"]:
        return "tsv"
    if ext in [".json", ".jsonl"]:
        return "json"
    # default attempt json
    return "json"


# ensure_lang_token removed; tokens are seeded at load time


def build_lora_model_if_requested(model, args: argparse.Namespace):
    if not args.use_lora:
        return model
    if not PEFT_AVAILABLE:
        print(
            "[WARN] PEFT not available. Install 'peft' to use LoRA. Proceeding without LoRA."
        )
        return model

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=(
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "fc1",
                "fc2",
            ]
        ),
    )
    model = get_peft_model(model, lora_config)
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer_name = args.tokenizer_name or args.model_name

    # Build additional_special_tokens: FAIRSEQ_LANGUAGE_CODES + LANG_GROUP_MAP values
    base_tokens: List[str] = list(FAIRSEQ_LANGUAGE_CODES)
    if PREPARE_LANG_GROUP_MAP is not None:
        base_tokens.extend(list(PREPARE_LANG_GROUP_MAP.values()))
    # Deduplicate while preserving order
    seen: set[str] = set()
    additional_tokens: List[str] = []
    for tok in base_tokens:
        if tok is None:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        additional_tokens.append(tok)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        additional_special_tokens=additional_tokens,
        src_lang="eng_Latn",
        tgt_lang="yue_Hant",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, attn_implementation="sdpa"
    )

    # Ensure model embeddings cover tokenizer size when tokens were added at load time
    try:
        if hasattr(model, "get_input_embeddings"):
            current = model.get_input_embeddings().num_embeddings
            if len(tokenizer) > current:
                model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load datasets and assert per-example language columns exist
    datasets = DatasetDict()
    datasets["train"] = load_dataset("json", data_files=args.train_file, split="train")
    datasets["to_zh_validation"] = load_dataset(
        "json", data_files=args.to_zh_validation_file, split="train"
    )
    datasets["to_formosan_validation"] = load_dataset(
        "json", data_files=args.to_formosan_validation_file, split="train"
    )

    for split_name, ds in datasets.items():
        if (
            args.source_column not in ds.column_names
            or args.target_column not in ds.column_names
        ):
            raise ValueError(
                f"Missing columns in {split_name} dataset. Expected: '{args.source_column}', '{args.target_column}'. "
                f"Found: {ds.column_names}"
            )
        if ("src_lang" not in ds.column_names) or ("tgt_lang" not in ds.column_names):
            raise ValueError(
                f"Missing 'src_lang'/'tgt_lang' in {split_name} dataset. Found: {ds.column_names}"
            )

    def preprocess_per_sample(example):
        inputs = example[args.source_column]
        targets = example[args.target_column]
        model_inputs = tokenizer(
            inputs,
            text_target=targets,
        )

        model_inputs["input_ids"][0] = tokenizer.convert_tokens_to_ids(
            example["src_lang"]
        )
        model_inputs["labels"][0] = tokenizer.convert_tokens_to_ids(example["tgt_lang"])
        return model_inputs

    datasets = datasets.map(preprocess_per_sample, num_proc=8)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=tokenizer.pad_token_id,
    )

    # Metric: BLEU via sacrebleu
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        tokenize = "13a"
        if labels[0][0] == tokenizer.convert_tokens_to_ids("zho_Hant"):
            tokenize = "zh"
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = (
            [decoded_labels]
            if isinstance(decoded_labels, str)
            else [[label.strip()] for label in decoded_labels]
        )
        result = sacrebleu.compute(
            predictions=decoded_preds, references=decoded_labels, tokenize=tokenize
        )
        result_dict = {"bleu": result["score"]}
        pred_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        result_dict["gen_len"] = (
            float(np.mean(pred_lens)) if len(pred_lens) > 0 else 0.0
        )
        return {k: round(v, 4) for k, v in result_dict.items()}

    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        load_best_model_at_end=True,
        metric_for_best_model="eval_to_zh_bleu",
        greater_is_better=True,
        report_to=None if args.report_to == "none" else [args.report_to],
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.get("train"),
        eval_dataset={
            "to_zh": datasets.get("to_zh_validation"),
            "to_formosan": datasets.get("to_formosan_validation"),
        },
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save
    trainer.save_state()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Record language handling
    lang_meta_path = os.path.join(args.output_dir, "lang_config.txt")
    try:
        with open(lang_meta_path, "w", encoding="utf-8") as f:
            f.write("per_example_lang=true\n")
            f.write("token_format=raw_codes\n")
    except Exception as e:
        print(f"[WARN] Failed to write lang_config.txt: {e}")


if __name__ == "__main__":
    main()
