"""Shared translation engine for the NLLB-200 English <-> Swahili model.

Usage:
    from engine import Translator

    translator = Translator("models/nllb-en-sw-ct2")  # or "models/nllb-en-sw"
    print(translator.translate("Hello.", "eng_Latn", "swh_Latn"))
"""
import os
import re
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANGUAGES = {"eng_Latn": "English", "swh_Latn": "Swahili"}
MAX_CHARS = 5000


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_sentences(text):
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            lines.append([])
            continue
        lines.append([p for p in re.split(r"(?<=[.!?])\s+", line) if p])
    return lines


class Translator:
    def __init__(self, model_path, backend="auto", num_beams=4):
        self.model_path = os.path.normpath(model_path)
        self.num_beams = num_beams
        self._lock = threading.Lock()
        if backend == "auto":
            backend = "ct2" if os.path.exists(os.path.join(self.model_path, "model.bin")) else "hf"
        if backend not in ("hf", "ct2"):
            raise ValueError(f"Unknown backend: {backend}")
        self.backend = backend
        self._load()

    @property
    def model_name(self):
        return os.path.basename(self.model_path)

    def _load(self):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.backend == "hf":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device).eval()
        else:
            import ctranslate2
            self._ct2 = ctranslate2.Translator(self.model_path, device="cpu", compute_type="int8")

    def _validate(self, src, tgt):
        if src not in LANGUAGES:
            raise ValueError(f"Unsupported language: {src}")
        if tgt not in LANGUAGES:
            raise ValueError(f"Unsupported language: {tgt}")
        if src == tgt:
            raise ValueError("Source and target languages must differ")

    def _translate_raw(self, texts, src, tgt):
        with self._lock:
            if self.backend == "hf":
                return self._translate_hf(texts, src, tgt)
            return self._translate_ct2(texts, src, tgt)

    def _translate_hf(self, texts, src, tgt):
        self.tokenizer.src_lang = src
        outputs = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt),
                    max_length=256,
                    num_beams=self.num_beams,
                )
            outputs.extend(self.tokenizer.batch_decode(out, skip_special_tokens=True))
        return outputs

    def _translate_ct2(self, texts, src, tgt):
        self.tokenizer.src_lang = src
        sources = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t, truncation=True, max_length=256))
            for t in texts
        ]
        results = self._ct2.translate_batch(
            sources,
            target_prefix=[[tgt]] * len(sources),
            beam_size=self.num_beams,
            max_batch_size=16,
        )
        outputs = []
        for r in results:
            tokens = r.hypotheses[0][1:]  # drop the target language tag
            outputs.append(self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True))
        return outputs

    def translate_batch(self, texts, src, tgt):
        self._validate(src, tgt)
        return self._translate_raw(texts, src, tgt)

    def translate(self, text, src, tgt):
        self._validate(src, tgt)
        if len(text) > MAX_CHARS:
            raise ValueError(f"Text too long (max {MAX_CHARS} characters)")
        if not text.strip():
            return ""
        lines = split_sentences(text)
        flat = [s for line in lines for s in line]
        translated = self._translate_raw(flat, src, tgt)
        rebuilt = []
        i = 0
        for line in lines:
            if not line:
                rebuilt.append("")
            else:
                rebuilt.append(" ".join(translated[i : i + len(line)]))
                i += len(line)
        return "\n".join(rebuilt)
