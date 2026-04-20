#!/usr/bin/env python3
"""Extract ids from <mt_start>...<mt_end> blocks."""

import argparse
import json
import re
import sys
from typing import List


MT_BLOCK_PATTERN = re.compile(r"(<mt_start>(?:<mt_\d{4}>)+<mt_end>)")
MT_ID_PATTERN = re.compile(r"<mt_(\d{4})>")


def extract_mt_ids(text: str) -> List[List[int]]:
    """Return all id lists found in mt blocks."""
    blocks = MT_BLOCK_PATTERN.findall(text)
    return [[int(token_id) for token_id in MT_ID_PATTERN.findall(block)] for block in blocks]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract <mt_xxxx> ids from <mt_start>...<mt_end> blocks."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Input text. If omitted, read from --input-file or stdin.",
    )
    parser.add_argument(
        "--input-file",
        help="Read input text from a file.",
    )
    parser.add_argument(
        "--keep-string-id",
        action="store_true",
        help="Keep ids as zero-padded strings instead of converting to integers.",
    )
    return parser


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.input_file is not None:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    text = load_text(args)

    blocks = MT_BLOCK_PATTERN.findall(text)
    if not blocks:
        print(json.dumps({"blocks": [], "id_lists": []}, ensure_ascii=False, indent=2))
        return 0

    if args.keep_string_id:
        id_lists = [MT_ID_PATTERN.findall(block) for block in blocks]
    else:
        id_lists = extract_mt_ids(text)

    print(
        json.dumps(
            {
                "blocks": blocks,
                "id_lists": id_lists,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
