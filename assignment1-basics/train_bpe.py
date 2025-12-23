from cs336_basics.tokenizer.tokenizer import train_bpe

TINY_STORIES = {
    "data_path": "data/TinyStoriesV2-GPT4-train.txt",
    "vocab_size": 10_000,
    "special_tokens": ["<|endoftext|>"],
    "save_dir": "./checkpoints/tiny_stories",
}

OWT = {
    "data_path": "data/owt_train.txt",
    "vocab_size": 32_000,
    "special_tokens": [],
    "save_dir": "./checkpoints/owt",
}
TINY_STORIES_PATH = "data/TinyStoriesV2-GPT4-train.txt"


if __name__ == "__main__":
    dataset = OWT

    train_bpe(
        dataset["data_path"],
        vocab_size=dataset["vocab_size"],
        special_tokens=dataset["special_tokens"],
        verbose=True,
        save_path=dataset["save_dir"],
    )
