#!/usr/bin/env bash
set -euo pipefail

# Example training script for finetuning NLLB
# Usage:
#   bash run_finetune.sh data/train.jsonl data/valid.jsonl eng_Latn yue_Hant outputs/nllb-yue

TRAIN_FILE=${1:-data.json}
TO_ZH_VALID_FILE=${2:-data_eval_to_zh.json}
TO_FORMOSAN_VALID_FILE=${3:-data_eval_to_formosan.json}
OUT_DIR=${4:-outputs/nllb-formosan-lr1e-4}
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  finetune_nllb.py \
  --model_name facebook/nllb-200-distilled-600M \
  --train_file "$TRAIN_FILE" \
  --to_zh_validation_file "$TO_ZH_VALID_FILE" \
  --to_formosan_validation_file "$TO_FORMOSAN_VALID_FILE" \
  --source_column src_text \
  --target_column tgt_text \
  --output_dir "$OUT_DIR" \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_source_length 512 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1000 \
  --report_to wandb \
  --learning_rate 1e-4 \
  --bf16


