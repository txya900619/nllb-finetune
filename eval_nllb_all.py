#!/usr/bin/env python3
import argparse
import csv
from typing import List

import evaluate
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def remove_heading_punctuation(text: str) -> str:
    if text[0] in ",，.。、：；!?！？：；":
        text = text[1:]
    return text


def replace_trailing_punctuation(text: str) -> str:
    if text[-1] in ",:":
        text = text[:-1] + "."
    if text[-1] in "，、：":
        text = text[:-1] + "。"
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate NLLB model and save CSV with BLEU"
    )
    p.add_argument(
        "--model_dir", required=True, help="Path or HF id of fine-tuned model"
    )
    p.add_argument("--dataset", required=True, help="HF dataset id")
    p.add_argument(
        "--split",
        default="validation",
        help="HF split to evaluate (default: validation)",
    )
    # Column names in the HF dataset
    p.add_argument("--source_column", default="transcript")
    p.add_argument("--target_column", default="translation")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=1)
    return p.parse_args()


def run_evaluation(model, ds, src_col, tgt_col, src_lang_code, tgt_lang_code, args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=True, src_lang=src_lang_code, tgt_lang=tgt_lang_code
    )
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    refs: List[List[str]] = []
    hyps: List[str] = []

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "forced_bos_token_id": tokenizer.convert_tokens_to_ids(tgt_lang_code),
    }

    # batched generation
    for start in tqdm(range(0, len(ds), args.batch_size)):
        batch = ds[start : start + args.batch_size]
        inputs = batch[src_col]
        enc = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        batch_hyps = tokenizer.batch_decode(out, skip_special_tokens=True)

        batch_refs = batch[tgt_col]

        for h, r in zip(batch_hyps, batch_refs):
            hyps.append(h)
            refs.append([r])

    bleu = sacrebleu.compute(
        predictions=hyps,
        references=refs,
        tokenize="zh" if tgt_col == "translation" else "13a",
        lowercase=True,
    )
    bleu_score = bleu.get("score", 0.0)

    chrf2 = chrf.compute(
        predictions=hyps,
        references=refs,
        word_order=2,
        lowercase=True,
    )
    chrf2_score = chrf2.get("score", 0.0)

    return bleu_score, chrf2_score


try:
    from prepare_data import LANG_GROUP_MAP as PREPARE_LANG_GROUP_MAP
except Exception:
    PREPARE_LANG_GROUP_MAP = None


def main():
    args = parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, attn_implementation="sdpa"
    )
    model.to("cuda")
    model.eval()

    to_zh_bleu_dict = {}
    to_zh_chrf2_dict = {}
    to_formosan_bleu_dict = {}
    to_formosan_chrf2_dict = {}

    for config in get_dataset_config_names(args.dataset):
        ds = load_dataset(args.dataset, name=config, split=args.split)
        ds = ds.remove_columns(
            [
                c
                for c in ds.column_names
                if c not in [args.source_column, args.target_column]
            ]
        )

        ds = ds.filter(
            lambda x: x[args.source_column] is not None
            and x[args.target_column] is not None
        )

        ds = ds.map(
            lambda x: {
                args.source_column: remove_heading_punctuation(x[args.source_column]),
                args.target_column: remove_heading_punctuation(x[args.target_column]),
            }
        )

        ds = ds.filter(
            lambda x: len(x[args.source_column]) > 0 and len(x[args.target_column]) > 0
        )

        ds = ds.map(
            lambda x: {
                args.source_column: replace_trailing_punctuation(x[args.source_column]),
                args.target_column: replace_trailing_punctuation(x[args.target_column]),
            }
        )

        # Validate columns exist
        for c in [args.source_column, args.target_column]:
            if c not in ds.column_names:
                raise ValueError(
                    f"Missing required column '{c}' in HF dataset. Found: {ds.column_names}"
                )

        # Determine language codes by config name
        if PREPARE_LANG_GROUP_MAP is None:
            raise RuntimeError(
                "prepare_data.LANG_GROUP_MAP not available to resolve language code from config"
            )
        lang_key = config.replace(" ", "_")
        indig_code = PREPARE_LANG_GROUP_MAP.get(lang_key)
        if indig_code is None:
            raise ValueError(
                f"Cannot resolve indigenous language code from config='{config}'. Check LANG_GROUP_MAP keys."
            )

        to_zh_bleu_score, to_zh_chrf2_score = run_evaluation(
            model,
            ds,
            args.source_column,
            args.target_column,
            indig_code,
            "zho_Hant",
            args,
        )

        to_formosan_bleu_score, to_formosan_chrf2_score = run_evaluation(
            model,
            ds,
            args.target_column,
            args.source_column,
            "zho_Hant",
            indig_code,
            args,
        )

        to_zh_bleu_dict[indig_code] = to_zh_bleu_score
        to_zh_chrf2_dict[indig_code] = to_zh_chrf2_score
        to_formosan_bleu_dict[indig_code] = to_formosan_bleu_score
        to_formosan_chrf2_dict[indig_code] = to_formosan_chrf2_score

    with open("to_zh_results_old.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["direction", "bleu", "chrf++"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "direction": f"{indig_code.lower()} → zh",
                    "bleu": to_zh_bleu_dict[indig_code],
                    "chrf++": to_zh_chrf2_dict[indig_code],
                }
                for indig_code in to_zh_bleu_dict.keys()
            ]
        )

    with open("to_formosan_results_old.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["direction", "bleu", "chrf++"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "direction": f"zh  → {indig_code.lower()}",
                    "bleu": to_formosan_bleu_dict[indig_code],
                    "chrf++": to_formosan_chrf2_dict[indig_code],
                }
                for indig_code in to_formosan_bleu_dict.keys()
            ]
        )


if __name__ == "__main__":
    import torch  # lazy import to keep startup fast

    main()
