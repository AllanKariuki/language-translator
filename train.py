"""Fine-tune NLLB-200-distilled-600M on English<->Swahili, bidirectionally.

Every pair is used in both directions (en->sw and sw->en), so --max-samples
100000 yields 200k training examples. The best checkpoint is picked by chrF on
the validation set, and training stops early once chrF stops improving.

Usage:
    python train.py --max-samples 100000 --epochs 1 --batch-size 8

Run prepare_data.py first. Output model: models/nllb-en-sw
"""
import argparse
import gc

import numpy as np
import sacrebleu
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_MODEL = "facebook/nllb-200-distilled-600M"
EN, SW = "eng_Latn", "swh_Latn"
MAX_LEN = 128


def tokenize_direction(dataset, tokenizer, src_key, tgt_key, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    def fn(batch):
        src = [t[src_key] for t in batch["translation"]]
        tgt = [t[tgt_key] for t in batch["translation"]]
        return tokenizer(src, text_target=tgt, max_length=MAX_LEN, truncation=True)

    return dataset.map(fn, batched=True, remove_columns=dataset.column_names)


def build_split(ds_split, tokenizer, max_samples=None):
    if max_samples:
        ds_split = ds_split.shuffle(seed=42).select(range(min(max_samples, len(ds_split))))
    # Every pair trains both en->sw and sw->en, so one model does both.
    en_sw = tokenize_direction(ds_split, tokenizer, "en", "sw", EN, SW)
    sw_en = tokenize_direction(ds_split, tokenizer, "sw", "en", SW, EN)
    return concatenate_datasets([en_sw, sw_en]).shuffle(seed=42)


class NllbTrainer(Seq2SeqTrainer):
    """Forces the right target language when generating during evaluation.

    Labels start with the target language token (e.g. swh_Latn), so we seed each
    row's decoder with [</s>, lang] — this works even when a batch mixes directions.
    """

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        if self.args.predict_with_generate and not prediction_loss_only:
            labels = inputs["labels"]
            start = torch.full_like(labels[:, :1], model.config.decoder_start_token_id)
            gen_kwargs = {
                **getattr(self, "_gen_kwargs", {}),
                **gen_kwargs,
                "decoder_input_ids": torch.cat([start, labels[:, :1]], dim=1),
            }
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)


class FreeMemoryCallback(TrainerCallback):
    """Return cached memory to the OS after evals/saves.

    On MPS, memory freed after a checkpoint save or eval stays cached by PyTorch,
    so usage ratchets up with every checkpoint until the Mac starts swapping.
    """

    def _free(self, *args, **kwargs):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    on_evaluate = on_save = _free


def make_compute_metrics(tokenizer):
    sw_id = tokenizer.convert_tokens_to_ids(SW)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        to_sw = labels[:, 0] == sw_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        hyps = tokenizer.batch_decode(preds, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = {}
        for name, mask in [("en_sw", to_sw), ("sw_en", ~to_sw)]:
            h = [x for x, m in zip(hyps, mask) if m]
            r = [x for x, m in zip(refs, mask) if m]
            if h:
                metrics[f"chrf_{name}"] = sacrebleu.corpus_chrf(h, [r]).score
                metrics[f"bleu_{name}"] = sacrebleu.corpus_bleu(h, [r]).score
        metrics["chrf"] = sacrebleu.corpus_chrf(hyps, [refs]).score
        return metrics

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100_000, help="pairs; each is used in both directions")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-samples", type=int, default=200, help="validation pairs (x2 directions)")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3, help="evals without chrF gain before stopping")
    parser.add_argument("--output", default="models/nllb-en-sw")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    ds = load_from_disk("data/en-sw")
    train_data = build_split(ds["train"], tokenizer, args.max_samples)
    eval_data = build_split(ds["validation"], tokenizer, args.eval_samples)
    print(f"Train: {len(train_data)}  Eval: {len(eval_data)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        # Adafactor + gradient checkpointing keep memory low enough for a 24GB Mac
        # (AdamW alone holds ~5GB of optimizer state for this model).
        optim="adafactor",
        gradient_checkpointing=True,
        warmup_steps=0.03,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        save_only_model=True,  # 2.4GB checkpoints instead of 7GB; can't resume mid-run
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_LEN,
        generation_num_beams=1,  # greedy keeps periodic evals fast; evaluate.py uses beams
        logging_steps=100,
        report_to="none",
        # No fp16: unstable on MPS. Trainer auto-selects mps/cuda/cpu.
    )

    trainer = NllbTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), FreeMemoryCallback()],
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
