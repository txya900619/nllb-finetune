#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, List

import evaluate
from datasets import load_dataset
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
        "--config", required=True, help="HF config name (decides indigenous language)"
    )
    p.add_argument(
        "--split",
        default="validation",
        help="HF split to evaluate (default: validation)",
    )
    # Column names in the HF dataset
    p.add_argument("--source_column", default="transcript")
    p.add_argument("--target_column", default="translation")
    # Direction control
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Evaluate Chinese->indigenous instead of indigenous->Chinese",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--out_csv", required=True, help="Output CSV path")
    return p.parse_args()


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

    ds = load_dataset(args.dataset, name=args.config, split=args.split)
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
            args.source_column: replace_trailing_punctuation(
                remove_heading_punctuation(x[args.source_column])
            ),
            args.target_column: replace_trailing_punctuation(
                remove_heading_punctuation(x[args.target_column])
            ),
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
    lang_key = args.config.replace(" ", "_")
    indig_code = PREPARE_LANG_GROUP_MAP.get(lang_key)
    if indig_code is None:
        raise ValueError(
            f"Cannot resolve indigenous language code from config='{args.config}'. Check LANG_GROUP_MAP keys."
        )

    if args.reverse:
        src_lang_code = "zho_Hant"
        tgt_lang_code = indig_code
        src_col = args.target_column
        tgt_col = args.source_column
    else:
        src_lang_code = indig_code
        tgt_lang_code = "zho_Hant"
        src_col = args.source_column
        tgt_col = args.target_column
        print(src_lang_code, tgt_lang_code)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, use_fast=True, src_lang=src_lang_code, tgt_lang=tgt_lang_code
    )

    sacrebleu = evaluate.load("sacrebleu")

    results: List[Dict[str, str]] = []
    bleu_corpus_refs: List[List[str]] = []
    bleu_corpus_hyps: List[str] = []

    # Single-target-language evaluation for this config/direction
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
        hyps = tokenizer.batch_decode(out, skip_special_tokens=True)

        refs = batch[tgt_col]
        srcs = batch[src_col]

        for s, h, r in zip(srcs, hyps, refs):
            results.append(
                {
                    "src_text": s,
                    "hypothesis": h,
                    "reference": r,
                }
            )
            bleu_corpus_hyps.append(h)
            bleu_corpus_refs.append([r])

    bleu = sacrebleu.compute(
        predictions=bleu_corpus_hyps,
        references=bleu_corpus_refs,
        lowercase=True,
        tokenize="13a" if args.reverse else "zh",
    )
    bleu_score = bleu.get("score", 0.0)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["src_text", "hypothesis", "reference", "src_lang", "tgt_lang"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Corpus BLEU: {bleu_score:.2f}")
    print(f"Saved CSV to: {args.out_csv}")


if __name__ == "__main__":
    import torch  # lazy import to keep startup fast

    main()
