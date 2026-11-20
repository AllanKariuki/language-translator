# Language Translator — English ⇄ Kiswahili (extensible to local languages)

Live text translation by fine-tuning **NLLB-200 (distilled, 600M)** on English–Swahili parallel data. Runs on Apple Silicon (MPS).

## Why NLLB-200

- Pretrained on 200 languages, already strong at Swahili (`swh_Latn`).
- Crucially for the roadmap: it already includes many East African languages — Kikuyu (`kik_Latn`), Luo (`luo_Latn`), Kamba (`kam_Latn`), Luganda (`lug_Latn`), Somali (`som_Latn`), Amharic (`amh_Ethi`) — so adding a new local language pair is mostly a data problem, not a modeling problem. Same scripts, different language codes.
- The 600M distilled variant fine-tunes comfortably on an M-series Mac.

## Datasets

| Dataset | Use | Size (en–sw) |
|---|---|---|
| `Rogendo/English-Swahili-Sentence-Pairs` | Train + validation + test | ~210k pairs |

`prepare_data.py` cleans this corpus and carves it into 96% train / 2% validation / 2% test. Evaluation runs on the held-out test split.

> Note: many tutorials point to `Helsinki-NLP/opus-100` for en-sw, but that dataset does **not** actually include Swahili. FLORES-200 is the standard MT benchmark but is currently gated / script-based on the Hub and awkward to load, so we evaluate on a held-out split instead. You can add FLORES later once you've accepted its terms on Hugging Face.

For expansion later: OPUS (CCMatrix, ParaCrawl), Tatoeba, and Masakhane community corpora (e.g. LAFAND-MT) for other African languages.

## Quickstart

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt

# 1. Download + clean data
python prepare_data.py

# 2. Fine-tune (start small; scale up with --max-samples)
python train.py --max-samples 100000 --epochs 1

# 3. Live translation REPL
python translate.py                # uses your fine-tuned model
python translate.py --base         # or the base NLLB model, no training needed

# 4. Benchmark (BLEU + chrF) on the held-out test split
python evaluate.py --base --limit 200   # baseline to beat
python evaluate.py
```

## Web demo and API

Convert the fine-tuned model to CTranslate2 int8 (much faster on CPU):

```bash
python convert_ct2.py
```

Run the Gradio demo (self-contained, no server needed):

```bash
python app.py
```

Run the FastAPI service:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Translate:

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "source_lang": "eng_Latn", "target_lang": "swh_Latn"}'
```

Send feedback:

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"source_text": "Hello.", "source_lang": "eng_Latn", "target_lang": "swh_Latn", "translation": "Habari.", "rating": "up"}'
```

Run tests:

```bash
python -m pytest
```

### Deploying the demo to Hugging Face Spaces

1. Convert the model and upload the `models/nllb-en-sw-ct2` directory to a Hugging Face model repo.
2. Create a Gradio Space and add `app.py`, `engine.py`, `feedback.py`, and `requirements.txt`.
3. Point the model path at the uploaded repo (via `--model` or the default in `app.py`) and commit.

## Files

- `prepare_data.py` — downloads the en-sw corpus, cleans (length ratio, dedupe), splits train/val/test, saves to `data/`
- `train.py` — bidirectional fine-tuning (en→sw and sw→en in one model) with HF `Seq2SeqTrainer`
- `translate.py` — interactive REPL; `:swap` flips direction, `:quit` exits
- `evaluate.py` — BLEU + chrF on the held-out test split, both directions

## Roadmap to local languages

1. Validate the pipeline on en⇄sw (this repo).
2. Pick the next language (e.g. Kikuyu). Gather parallel data: Masakhane, OPUS, Bible corpora, community translation drives.
3. Fine-tune the same checkpoint with the new pair added — NLLB is multilingual, one model serves all pairs.
4. For truly low-resource languages with no NLLB support, use sw or en as a pivot and back-translation to synthesize data.
5. Later: wrap in FastAPI for streaming "live" translation; add Whisper (ASR) + TTS for speech.

## Hardware notes (Apple Silicon)

Training auto-uses the `mps` device. Keep `--batch-size 8` with gradient accumulation; avoid fp16 on MPS. A 100k-sample epoch takes a few hours on an M5 Pro. For bigger runs, rent a cloud GPU and the same scripts work on CUDA unchanged.
