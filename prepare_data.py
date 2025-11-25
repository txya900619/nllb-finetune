#!/usr/bin/env python3
r"""Convert a Hugging Face translation dataset (all configs) to LLaMA-Factory ShareGPT format.

This script targets MT-style data where `transcript` (source in an indigenous language)
and `translation` (target in Mandarin) are provided. It emits messages using the prompt:

  Translate the following segment into <target_language>, without additional explanation.

  <source_text>

- sharegpt format (default):
  [
    {
      "messages": [
        {"role": "system", "content": "You are a helpful machine translation assistant."},
        {"role": "user", "content": "Translate the following segment into Chinese, without additional explanation.\n\n<src>"},
        {"role": "assistant", "content": "<tgt>"}
      ]
    }
  ]

Usage examples:
  python prepare_mt_data.py \
    --dataset your_org/your_dataset --src_col transcript --tgt_col translation \
    --split train --num_proc 8 --out data/mt_sharegpt.json

This script iterates over all configs of the dataset automatically.
"""

import argparse
import json
import re
from typing import Any, Optional

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

LANG_GROUP_MAP = {
    "Amis_Coastal": "ami_Coas",
    "Amis_Hengchun": "ami_Heng",
    "Amis_Malan": "ami_Mala",
    "Amis_Southern": "ami_Sout",
    "Amis_Xiuguluan": "ami_Xiug",
    "Atayal_FourSeasons": "tay_Four",
    "Atayal_Sekolik": "tay_Seko",
    "Atayal_Wanda": "tay_Wand",
    "Atayal_Wenshui": "tay_Wens",
    "Atayal_YilanZeaol": "tay_Yzea",
    "Atayal_Zeaol": "tay_Zeao",
    "Bunun_Junqun": "bnn_Junq",
    "Bunun_Kaqun": "bnn_Kaqu",
    "Bunun_Luanqun": "bnn_Luan",
    "Bunun_Tanqun": "bnn_Tanq",
    "Bunun_Zhuoqun": "bnn_Zhuo",
    "Kanakanavu": "xnb_Kana",
    "Kavalan": "ckv_Kava",
    "Paiwan_Central": "pwn_Cent",
    "Paiwan_Eastern": "pwn_East",
    "Paiwan_Northern": "pwn_Nrth",
    "Paiwan_Southern": "pwn_Sout",
    "Puyuma_Jianhe": "pyu_Jian",
    "Puyuma_Nanwang": "pyu_Nanw",
    "Puyuma_Xiqun": "pyu_Xiqu",
    "Puyuma_Zhiben": "pyu_Zhib",
    "Rukai_Dawu": "dru_Dawu",
    "Rukai_Dona": "dru_Dona",
    "Rukai_Eastern": "dru_East",
    "Rukai_Maolin": "dru_Maol",
    "Rukai_Wanshan": "dru_Wans",
    "Rukai_Wutai": "dru_Wuta",
    "Saaroa": "sxr_Saar",
    "Saisiyat": "xsy_Sais",
    "Sakizaya": "szy_Saki",
    "Seediq_DeluValley": "trv_Delu",
    "Seediq_Duda": "trv_Duda",
    "Seediq_Tegudaya": "trv_Tegu",
    "Thao": "ssf_Thao",
    "Truku": "trv_Truk",
    "Tsou": "tsu_Tsou",
    "Yami": "tao_Yami",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HF MT datasets to LLaMA-Factory ShareGPT format"
    )

    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="HF dataset name or path (repeatable: pass multiple --dataset)",
    )
    parser.add_argument(
        "--src_col",
        default="transcript",
        help="Source text column name (default: transcript)",
    )
    parser.add_argument(
        "--tgt_col",
        default="translation",
        help="Target text column name (default: translation)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to load for all datasets (default: train)",
    )
    parser.add_argument(
        "--lang_group_col",
        default="lang_group_en",
        help="Column containing indigenous language English name; underscores will be replaced by spaces",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional max samples per dataset (for quick tests)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for parallel filter/map (default: datasets default)",
    )
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument(
        "--config-filter", help="Optional config name filter (substring match)"
    )

    return parser.parse_args()


def get_all_configs(dataset_name: str) -> list[str]:
    try:
        return get_dataset_config_names(dataset_name) or [None]
    except Exception:
        return [None]


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


def prepare_config_datasets(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    src_col: str,
    tgt_col: str,
    lang_group_col: Optional[str],
    max_samples: Optional[int],
    num_proc: Optional[int],
):
    ds = load_dataset(dataset_name, name=config_name, split=split)

    # Keep only needed flat columns
    columns = set(ds.column_names)
    keep_cols = {src_col, tgt_col}
    keep_cols.add("id" if "id" in columns else None)
    if lang_group_col and (lang_group_col in columns):
        keep_cols.add(lang_group_col)
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    # Filter rows missing required flat fields
    def _has_required(ex: dict) -> bool:
        s = ex.get(src_col)
        t = ex.get(tgt_col)

        if not isinstance(s, str) or not isinstance(t, str):
            return False

        s = re.sub(r"[,，.。、：；!?！？：；]", "", s)
        t = re.sub(r"[,，.。、：；!?！？：；]", "", t)

        return len(s.strip()) > 0 and len(t.strip()) > 0

    ds = ds.filter(_has_required, num_proc=num_proc)

    def _not_word(ex: dict) -> bool:
        return "word" not in ex["id"]

    if "id" in columns:
        ds = ds.filter(_not_word, num_proc=num_proc)

    # Forward: indigenous -> target_language
    def _map_forward(ex: dict) -> dict[str, Any]:
        src_text = ex[src_col].strip()
        tgt_text = ex[tgt_col].strip()

        src_text = remove_heading_punctuation(src_text)
        tgt_text = remove_heading_punctuation(tgt_text)

        src_text = replace_trailing_punctuation(src_text)
        tgt_text = replace_trailing_punctuation(tgt_text)

        return {
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_lang": LANG_GROUP_MAP["_".join(ex[lang_group_col].split(" "))],
            "tgt_lang": "zho_Hant",
        }

    forward_ds = ds.map(_map_forward, remove_columns=ds.column_names, num_proc=num_proc)
    if max_samples is not None and len(forward_ds) > max_samples:
        forward_ds = forward_ds.select(range(max_samples))

    # Reverse: Chinese -> indigenous (requires language name)
    reverse_ds = None
    if lang_group_col and (lang_group_col in columns):

        def _has_lang(ex: dict) -> bool:
            raw = ex.get(lang_group_col)
            return isinstance(raw, str) and len(raw) > 0

        ds_rev = ds.filter(_has_lang, num_proc=num_proc)

        def _map_reverse(ex: dict) -> dict[str, Any]:
            src_text = ex[tgt_col].strip()
            tgt_text = ex[src_col].strip()

            src_text = remove_heading_punctuation(src_text)
            tgt_text = remove_heading_punctuation(tgt_text)

            src_text = replace_trailing_punctuation(src_text)
            tgt_text = replace_trailing_punctuation(tgt_text)

            return {
                "src_text": src_text,
                "tgt_text": tgt_text,
                "src_lang": "zho_Hant",
                "tgt_lang": LANG_GROUP_MAP["_".join(ex[lang_group_col].split(" "))],
            }

        reverse_ds = ds_rev.map(
            _map_reverse, remove_columns=ds_rev.column_names, num_proc=num_proc
        )
        if max_samples is not None and len(reverse_ds) > max_samples:
            reverse_ds = reverse_ds.select(range(max_samples))

    return forward_ds, reverse_ds


def main() -> None:
    args = parse_args()

    dataset_parts = []
    for dataset_name in args.dataset:
        for config_name in get_all_configs(dataset_name):
            if args.config_filter and (args.config_filter not in config_name):
                continue

            fwd, rev = prepare_config_datasets(
                dataset_name,
                config_name,
                args.split,
                args.src_col
                if dataset_name
                not in ["ithuan/ithuan_formosan_text", "ithuan/formosan_bible"]
                else "formosan",
                args.tgt_col
                if dataset_name
                not in ["ithuan/ithuan_formosan_text", "ithuan/formosan_bible"]
                else "mandarin",
                args.lang_group_col,
                args.max_samples,
                args.num_proc,
            )

            if fwd is not None and len(fwd) > 0:
                dataset_parts.append(fwd)
            if rev is not None and len(rev) > 0:
                dataset_parts.append(rev)

    out_records: list[dict[str, Any]] = []
    if len(dataset_parts) > 0:
        merged = concatenate_datasets(dataset_parts)
        out_records = merged.to_list()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
