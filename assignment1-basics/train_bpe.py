from cs336_basics.tokenizer import train_bpe

TINY_STORIES_PATH = "data/TinyStoriesV2-GPT4-train.txt"


if __name__ == "__main__":
    train_bpe(
        TINY_STORIES_PATH,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        verbose=True,
        save_path="./checkpoints/tiny_stories",
    )
