"""Benchmark on the held-out test split: BLEU + chrF, both directions.

Usage:
    python evaluate.py                 # fine-tuned model
    python evaluate.py --base          # base NLLB (your baseline to beat)
    python evaluate.py --limit 200     # quicker run

Run prepare_data.py first (creates data/en-sw with a 'test' split).
"""
import argparse
import os

import sacrebleu
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from engine import Translator

BASE_MODEL = "facebook/nllb-200-distilled-600M"
EN, SW = "eng_Latn", "swh_Latn"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def translate_batch(texts, model, tokenizer, src_code, tgt_code, device, batch_size=16):
    tokenizer.src_lang = src_code
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=256,
                num_beams=4,
            )
        outputs.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/nllb-en-sw")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--limit", type=int, default=500, help="sentences per direction")
    args = parser.parse_args()

    path = BASE_MODEL if args.base else args.model
    use_ct2 = not args.base and os.path.exists(os.path.join(path, "model.bin"))

    if use_ct2:
        print(f"Loading {path} (ctranslate2 int8) ...")
        translator = Translator(path)

        def translate(texts, src_code, tgt_code):
            return translator.translate_batch(texts, src_code, tgt_code)

    else:
        device = get_device()
        print(f"Loading {path} on {device} ...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()

        def translate(texts, src_code, tgt_code):
            return translate_batch(texts, model, tokenizer, src_code, tgt_code, device)

    test = load_from_disk("data/en-sw")["test"]
    if args.limit:
        test = test.select(range(min(args.limit, len(test))))
    en = [ex["translation"]["en"] for ex in test]
    sw = [ex["translation"]["sw"] for ex in test]

    for name, src_texts, refs, src_code, tgt_code in [
        ("en->sw", en, sw, EN, SW),
        ("sw->en", sw, en, SW, EN),
    ]:
        print(f"\nTranslating {name} ({len(src_texts)} sentences) ...")
        hyps = translate(src_texts, src_code, tgt_code)
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        print(f"{name}:  BLEU {bleu.score:.2f}   chrF {chrf.score:.2f}")


if __name__ == "__main__":
    main()
