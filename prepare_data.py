"""Download and clean English<->Swahili parallel data, save to data/en-sw.

Source: Rogendo/English-Swahili-Sentence-Pairs (~210k pairs, verified on HF Hub).
(opus-100 does NOT include Swahili, despite older tutorials suggesting it.)

Creates train/validation/test splits in the same {"translation": {"en", "sw"}}
format that train.py and evaluate.py expect.

Usage: python prepare_data.py [--max-len 256] [--ratio 3.0]
"""
import argparse

from datasets import load_dataset

DATASET = "Rogendo/English-Swahili-Sentence-Pairs"
EN_COL = "English sentence"
SW_COL = "Swahili Translation"


def to_translation(ds):
    """Reshape flat columns into a single 'translation' dict column."""

    def fn(ex):
        return {"translation": {"en": (ex[EN_COL] or "").strip(),
                                "sw": (ex[SW_COL] or "").strip()}}

    return ds.map(fn, remove_columns=ds.column_names)


def clean(dataset, max_len: int, ratio: float):
    """Drop empty, overly long, or badly length-mismatched pairs; dedupe."""
    seen = set()

    def keep(ex):
        en = ex["translation"]["en"]
        sw = ex["translation"]["sw"]
        if not en or not sw:
            return False
        if len(en) > max_len * 4 or len(sw) > max_len * 4:  # chars, rough cap
            return False
        len_ratio = max(len(en), len(sw)) / max(min(len(en), len(sw)), 1)
        if len_ratio > ratio:
            return False
        key = (en.lower(), sw.lower())
        if key in seen:
            return False
        seen.add(key)
        return True

    return dataset.filter(keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=256, help="max length in tokens (approx)")
    parser.add_argument("--ratio", type=float, default=3.0, help="max source/target length ratio")
    args = parser.parse_args()

    print(f"Downloading {DATASET} ...")
    raw = load_dataset(DATASET, split="train")
    raw = to_translation(raw)

    before = len(raw)
    raw = clean(raw, args.max_len, args.ratio)
    print(f"Cleaned: {before} -> {len(raw)} pairs")

    # 96% train / 2% validation / 2% test
    split1 = raw.train_test_split(test_size=0.04, seed=42)
    split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
    ds = {
        "train": split1["train"],
        "validation": split2["train"],
        "test": split2["test"],
    }
    from datasets import DatasetDict
    ds = DatasetDict(ds)
    for split in ds:
        print(f"  {split}: {len(ds[split])}")

    ds.save_to_disk("data/en-sw")
    print("Saved to data/en-sw")


if __name__ == "__main__":
    main()
