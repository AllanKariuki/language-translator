"""Live translation REPL (English <-> Kiswahili).

Usage:
    python translate.py                     # fine-tuned model (models/nllb-en-sw)
    python translate.py --base              # base NLLB, works without training
    python translate.py --direction sw-en

Commands inside the REPL: :swap (flip direction), :quit
"""
import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "facebook/nllb-200-distilled-600M"
CODES = {"en": "eng_Latn", "sw": "swh_Latn"}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def translate(text, model, tokenizer, src, tgt, device):
    tokenizer.src_lang = CODES[src]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(CODES[tgt]),
            max_length=256,
            num_beams=4,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/nllb-en-sw")
    parser.add_argument("--base", action="store_true", help="use base NLLB model")
    parser.add_argument("--direction", default="en-sw", choices=["en-sw", "sw-en"])
    args = parser.parse_args()

    path = BASE_MODEL if args.base else args.model
    device = get_device()
    print(f"Loading {path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()

    src, tgt = args.direction.split("-")
    print(f"Ready. Translating {src} -> {tgt}. Commands: :swap, :quit\n")

    while True:
        try:
            text = input(f"[{src}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == ":quit":
            break
        if text == ":swap":
            src, tgt = tgt, src
            print(f"Now translating {src} -> {tgt}")
            continue
        print(f"[{tgt}]   {translate(text, model, tokenizer, src, tgt, device)}\n")


if __name__ == "__main__":
    main()
