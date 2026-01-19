import json
import os

DATA_PATH = {
    "math": "./data/math/data",
    "gsm8k": "./data/gsm8k",
    "mmlu": "./data/mmlu",
}

SAVED_DIR = "./data/pre-processed"


def save_jsonl(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_math(data_path: str):
    from cs336_alignment.dataset_utils.math import collect_rows, process_row

    train_rows = collect_rows(data_path, filename="train-00000-of-00001.parquet")
    test_rows = collect_rows(data_path, filename="test-00000-of-00001.parquet")

    train = [{"question": p, "cot": c, "answer": a} for p, c, a in (process_row(r) for r in train_rows)]
    test = [{"question": p, "cot": c, "answer": a} for p, c, a in (process_row(r) for r in test_rows)]
    return train, test


def process_gsm8k(data_path: str):
    from cs336_alignment.dataset_utils.gsm8k import collect_rows, process_row

    train_rows = collect_rows(data_path, filename="train.jsonl")
    test_rows = collect_rows(data_path, filename="test.jsonl")

    train = [{"question": p, "cot": c, "answer": a} for p, c, a in (process_row(r) for r in train_rows)]
    test = [{"question": p, "cot": c, "answer": a} for p, c, a in (process_row(r) for r in test_rows)]
    return train, test


def process_mmlu(data_path: str):
    from cs336_alignment.dataset_utils.mmlu import collect_rows

    # Use dev as train, test as test (common setup)
    train_rows = collect_rows(os.path.join(data_path, "dev"))
    test_rows = collect_rows(os.path.join(data_path, "test"))

    return train_rows, test_rows


PROCESSORS = {
    "math": process_math,
    "gsm8k": process_gsm8k,
    "mmlu": process_mmlu,
}


def main():
    os.makedirs(SAVED_DIR, exist_ok=True)

    for name, path in DATA_PATH.items():
        processor = PROCESSORS.get(name)
        if processor is None:
            raise ValueError(f"Unknown dataset: {name}")

        train, test = processor(path)

        out_dir = os.path.join(SAVED_DIR, name)
        save_jsonl(train, os.path.join(out_dir, "train.jsonl"))
        save_jsonl(test, os.path.join(out_dir, "test.jsonl"))

        print(f"[{name}] train={len(train)} test={len(test)} -> {out_dir}")


if __name__ == "__main__":
    main()
