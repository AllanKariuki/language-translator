# Changelog

This changelog records changes to the pipeline and the result of each training and evaluation stage. Newest entries are first.

Unless noted otherwise, every score is on the held-out en–sw test split, measured with `python evaluate.py --limit 200` (the first 200 sentences per direction, beam 4). With only 200 sentences, differences of about 2 BLEU or less are noise.

---

## 2026-11-20: First fine-tuned model ([PR #1](https://github.com/AllanKariuki/language-translator/pull/1))

### Fixed (`train.py`)
- **MPS out-of-memory crash.** Live tensors were only about 8 GiB, but PyTorch's MPS cache grew to about 21 GiB of "other allocations" and hit the 30 GiB cap. `FreeMemoryCallback` emptied the cache only after evals and saves, and the first eval was at step 250. It now also runs `on_step_end`, after every optimizer step.
- **MPS cache churn.** The data collator now pads to a multiple of 8 (`pad_to_multiple_of=8`), so batch shapes repeat and cached buffers get reused.
- **Logged training loss inflated by the gradient-accumulation factor.** M2M100's `forward` accepts `**kwargs`, so `Trainer` assumed the model normalised the loss itself and skipped dividing by the accumulation steps. M2M100 ignores `num_items_in_batch`, so the loss was never divided. `NllbTrainer` now sets `model_accepts_loss_kwargs = False`.

### Changed (`train.py`)
- Added `--grad-accum` (default 8) and lowered the default `--batch-size` from 8 to 4. The effective batch size stays 32.

### Training runs: `python train.py --max-samples 20000 --eval-steps 250`
Each run used 20k pairs in both directions (40k examples), 1 epoch, 1250 steps.

| Run | Outcome | Cause |
|---|---|---|
| 1 | Crashed at step 31: `MPS backend out of memory` | MPS cache growth. First suspected batch size, so batch went to 4 with accumulation 8. |
| 2 | Crashed at step 143: same error | The batch size change didn't help. The failed allocation (1000.8 MiB) is the shared embedding gradient (256,206 × 1024 × fp32), which doesn't depend on batch size. Logged loss was 8.48 at step 100, which revealed the loss-scaling bug. |
| 3 | Stopped at step 488 | No crash and memory stayed flat. The Mac went into deep sleep on battery for about 75 minutes, and Claude Code's 2-hour limit on background tasks then ended the run. Loss at step 100 was 1.064, confirming the loss fix. |
| 4 | **Completed** in 7,584 s (about 2 h 6 min) | Ran detached with `nohup` and kept awake with `caffeinate -dims -w <pid>`. Average training loss was 0.901. |

### Run 4 validation evals
Greedy decoding, 200 validation pairs per direction. These aren't directly comparable to the test-set scores.

| Step | Eval loss | en→sw BLEU / chrF | sw→en BLEU / chrF | Overall chrF |
|---|---|---|---|---|
| 250 | 0.9402 | 42.59 / 64.43 | 47.31 / 62.72 | 63.58 |
| 500 | 0.9176 | 41.57 / 63.57 | 47.51 / 62.58 | 63.08 |
| 750 | 0.9045 | 44.16 / 65.28 | 47.18 / 62.58 | 63.94 |
| 1000 | 0.8871 | 43.97 / 65.18 | 48.05 / 63.14 | 64.17 |
| 1250 | 0.8860 | 45.20 / 65.96 | 47.82 / 63.17 | **64.57** (best) |

The best checkpoint (step 1250) was saved to `models/nllb-en-sw`. Eval loss was still falling at the end, so a larger run should improve further.

### Test-set results: fine-tuned vs. base

| Direction | Base NLLB-600M BLEU / chrF | Fine-tuned BLEU / chrF | Change |
|---|---|---|---|
| en→sw | 39.32 / 63.66 | **46.07 / 70.15** | +6.75 / +6.49 |
| sw→en | 42.75 / 61.92 | **49.29 / 66.37** | +6.54 / +4.45 |

The test split comes from the same corpus as the training data, so these gains may overstate real-world improvement.

### Housekeeping
- Deleted the intermediate checkpoints (`checkpoint-1000`, `checkpoint-1250`). Only the final model is kept, which takes about 2.3 GB instead of about 7 GB.

### Notes for future runs
- Run training on AC power with the lid open. Use `caffeinate`, and launch with `nohup` if the run may take longer than 2 hours.
- Checkpoints are saved model-only (`save_only_model=True`), so a stopped run can't be resumed and has to restart.
- Close Chrome, Docker and other memory-heavy apps first. Chrome pushed swap to about 2.8 GB during run 4, but step speed was not affected.

---

## 2026-11-19: Initial commit

- NLLB-200-distilled-600M English⇄Swahili pipeline: `prepare_data.py`, `train.py`, `evaluate.py`, `translate.py`.
- Data: `Rogendo/English-Swahili-Sentence-Pairs`, split 96% train / 2% validation / 2% test.
- Training setup: bidirectional (each pair is used en→sw and sw→en), Adafactor, gradient checkpointing, early stopping on validation chrF.

---

## 2026-10-30: Baseline

Base `facebook/nllb-200-distilled-600M`, measured with `python evaluate.py --base --limit 200`:

| Direction | BLEU | chrF |
|---|---|---|
| en→sw | 39.32 | 63.66 |
| sw→en | 42.75 | 61.92 |

---

## 2026-07-24: First training attempt (failed)

- The run stopped before its first checkpoint, so no model was produced.
- Likely cause: system memory pressure. Chrome, Docker, VS Code and other apps kept 20–48 GB in swap, and training slowed from about 2 s per step to over 12 s or stalled. An earlier AdamW setup without cache freeing used 26–30 GB, which led to the switch to Adafactor, gradient checkpointing and `FreeMemoryCallback`.
