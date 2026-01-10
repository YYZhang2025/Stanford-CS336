import os

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer.tokenizer import (
    encode_file_to_bin,
    load_tokenizer_from_dir,
    train_bpe,
)

TINY_STORIES = {
    "train_data_path": "data/TinyStoriesV2-GPT4-train.txt",
    "dev_data_path": "data/TinyStoriesV2-GPT4-valid.txt",
    "vocab_size": 10_000,
    "special_tokens": ["<|endoftext|>"],
    "save_dir": "./datasets/tiny_stories",
}

OWT = {
    "train_data_path": "data/owt_train.txt",
    "vocab_size": 32_000,
    "special_tokens": [],
    "save_dir": "./datasets/owt",
}


if __name__ == "__main__":
    dataset = TINY_STORIES

    if not os.path.exists(dataset["save_dir"]):
        os.makedirs(dataset["save_dir"])
        train_bpe(
            dataset["train_data_path"],
            vocab_size=dataset["vocab_size"],
            special_tokens=dataset["special_tokens"],
            verbose=True,
            save_path=dataset["save_dir"],
        )

        print(f"BPE tokenizer trained and saved to {dataset['save_dir']}")
    elif os.path.exists(os.path.join(dataset["save_dir"], "vocab.json")) and os.path.exists(
        os.path.join(dataset["save_dir"], "merges.txt")
    ):
        print(f"Tokenizer already exists at {dataset['save_dir']}, skipping training.")
    else:
        print(f"Save directory {dataset['save_dir']} exists but tokenizer files not found. Training...")
        train_bpe(
            dataset["train_data_path"],
            vocab_size=dataset["vocab_size"],
            special_tokens=dataset["special_tokens"],
            verbose=True,
            save_path=dataset["save_dir"],
        )

    # Pre-tokenize the dataset
    tokenizer = load_tokenizer_from_dir(dataset["save_dir"])

    out_bin_path = os.path.join(dataset["save_dir"], "train.bin")
    encode_file_to_bin(tokenizer, dataset["train_data_path"], out_bin_path, dtype=np.uint16)
    print(f"Encoded training data saved to {out_bin_path}")

    out_bin_eval_path = os.path.join(dataset["save_dir"], "eval.bin")
    encode_file_to_bin(tokenizer, dataset["dev_data_path"], out_bin_eval_path, dtype=np.uint16)
    print(f"Encoded evaluation data saved to {out_bin_eval_path}")
