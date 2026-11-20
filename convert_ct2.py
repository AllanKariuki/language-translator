"""Convert the fine-tuned HF model to a CTranslate2 int8 model.

Usage:
    python convert_ct2.py [--model models/nllb-en-sw] [--output models/nllb-en-sw-ct2] [--force]
"""
import argparse
import os

import ctranslate2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/nllb-en-sw")
    parser.add_argument("--output", default="models/nllb-en-sw-ct2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    copy_files = [
        f
        for f in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "generation_config.json",
        )
        if os.path.exists(os.path.join(args.model, f))
    ]
    converter = ctranslate2.converters.TransformersConverter(args.model, copy_files=copy_files)
    converter.convert(args.output, quantization="int8", force=args.force)

    size = sum(
        os.path.getsize(os.path.join(dirpath, name))
        for dirpath, _, names in os.walk(args.output)
        for name in names
    )
    print(f"Converted {args.model} -> {args.output} ({size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
