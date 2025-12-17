import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


### --------- Helper functions --------------
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def string_to_bytes(s: str, return_int: bool = True) -> list[int] | list[bytes]:
    byte_array = s.encode("utf-8")
    return list(map(int, byte_array)) if return_int else list(map(bytes, byte_array))


def utf8_bytes_to_string(byte_indices: list[bytes]) -> str:
    return b"".join(byte_indices).decode("utf-8")


def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # idx -> byte representation
    current_index = 256

    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            vocab[current_index] = token_bytes
            current_index += 1

    return vocab


### --------- End helper functions --------------


### --------- Pre-process & Pre-tokenization steps --------------
# 1. Split by special tokens


def split_by_special_tokens(text: str, special_tokens: list[str], include_special: bool = False) -> list[str]:
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if include_special:
        special_chunks = re.split(f"({pattern})", text)
    else:
        # Split without capturing the special tokens
        special_chunks = re.split(pattern, text)

    return special_chunks


# 2. Split by regex pattern
def pre_tokenize(
    string: str, special_tokens: list[str], including_special: bool = False
) -> dict[tuple[bytes] | tuple[int], int]:
    word_counter = Counter()
    chunks = split_by_special_tokens(string, special_tokens, include_special=including_special)

    for chunk in chunks:
        if including_special and chunk in special_tokens:
            word_counter[tuple(string_to_bytes(chunk, return_int=False))] += 1
        else:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                word_counter[tuple(string_to_bytes(word))] += 1

    return word_counter


def pre_tokenize_string_worker(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    queue: Queue,
    start: int,
    end: int,
    include_special: bool = False,
):
    """
    Pre-tokenize a string into bytes.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    word_counter = pre_tokenize(chunk, special_tokens, include_special)

    # Put the result in the queue
    queue.put(word_counter)


### --------- End Pre-process steps --------------


@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]

    def save(self, file_dir: str):
        vocab_file = os.path.join(file_dir, "vocab.json")
        merges_file = os.path.join(file_dir, "merges.txt")
        import json

        with open(vocab_file, "w", encoding="utf-8") as vf:
            json.dump({str(k): list(v) for k, v in self.vocab.items()}, vf, ensure_ascii=False, indent=4)
        with open(merges_file, "w", encoding="utf-8") as mf:
            for (a, b), _ in sorted(self.merges.items(), key=lambda item: item[1]):
                mf.write(f"{a} {b}\n")


from multiprocessing import Manager, Process, Queue
from queue import Empty


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 1. Pre-tokenization
    # 1.1 Find chunk boundaries
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, desired_num_chunks=kwargs.get("desired_num_chunks", 16), split_special_token=b"\n"
        )

    # 1.2 Count word frequencies across chunks using multiprocessing
    manager = Manager()
    queue = manager.Queue()
    processes: list[Process] = []

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize,
            args=(
                input_path,
                start,
                end,
                special_tokens or [],
                kwargs.get("including_special", False),
                queue,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    word_counter = Counter()
    for _ in range(len(processes)):
        try:
            partial_counter = queue.get(timeout=10)
            word_counter.update(partial_counter)
        except Empty:
            continue

    # 2. BPE Merging
    pairs_freqs = get_stats(word_counter)
    for _ in range(num_merges):
        most_common_pair = get_most_frequent_pair(pairs_freqs)
        print(
            "Most common pair:",
            (vocab[most_common_pair[0]], vocab[most_common_pair[1]]),
            "->",
            pairs_freqs[most_common_pair],
        )

        new_index = add_pair_to_vocab(vocab, most_common_pair)

        merges[most_common_pair] = new_index

        word_counter, pairs_freqs = merge_pair_ids(word_counter, most_common_pair, new_index)

    return vocab, merges
