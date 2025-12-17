import os
from collections import Counter, defaultdict
from multiprocessing import Manager, Process, Queue
from queue import Empty
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = max(1, (os.cpu_count() or 1) - 4)


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


def string_to_bytes(s: str, return_int: bool = False) -> list[int] | list[bytes]:
    byte_array = s.encode("utf-8")
    return list(map(int, byte_array)) if return_int else [bytes([b]) for b in byte_array]


def utf8_bytes_to_string(byte_indices: list[bytes]) -> str:
    return b"".join(byte_indices).decode("utf-8")


def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # idx -> byte representation
    current_index = 256

    if special_tokens is not None:
        for token in special_tokens:
            vocab[current_index] = token.encode("utf-8")
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
def pre_tokenize(string: str, special_tokens: list[str], including_special: bool = False) -> Counter:
    word_counter = Counter()

    chunks = split_by_special_tokens(string, special_tokens, include_special=including_special)

    for chunk in chunks:
        if including_special and chunk in special_tokens:
            word_counter[tuple(string_to_bytes(chunk))] += 1
        else:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                word_counter[tuple(string_to_bytes(word, return_int=True))] += 1
    return word_counter


def pre_tokenize_string_worker(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    queue: Queue,
    start: int,
    end: int,
    include_special: bool = False,
):
    # Read the chunk from the file
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    word_counter = pre_tokenize(chunk, special_tokens, include_special)

    # Put the result in the queue
    queue.put(word_counter)


### --------- End Pre-process steps --------------
def pair_counts(
    word_counter: dict[tuple[int, ...], int],
) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}

    for token, freq in word_counter.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq

    return pairs


def get_most_frequent_pair(
    pair_counter: dict[tuple[int, int], int], vocab: dict[int, bytes]
) -> tuple[int, int]:
    """
    If the most frequent pair is not unique, return the one with the highest
    byte representation in lexicographical order.
    """
    max_freq = max(pair_counter.values())

    candidates = [
        (pair, (vocab[pair[0]], vocab[pair[1]])) for pair, freq in pair_counter.items() if freq == max_freq
    ]
    candidates.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)

    return candidates[0][0]


def add_pair_to_vocab(
    vocab: dict[int, bytes],
    pair: tuple[int, int],
) -> int:
    index = len(vocab)
    vocab[index] = vocab[pair[0]] + vocab[pair[1]]

    return index


def merge_pairs(
    word_counter: dict[tuple[int], int],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int], int], dict[tuple[int, int], int]]:
    new_word_counter: defaultdict[tuple[int], int] = defaultdict(int)
    updated_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    for word, freq in word_counter.items():
        new_word = []
        i = 0

        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word_counter[tuple(new_word)] += freq

        for index1, index2 in zip(new_word[:-1], new_word[1:]):
            updated_pair_counts[(index1, index2)] = updated_pair_counts.get((index1, index2), 0) + freq

    return new_word_counter, updated_pair_counts


def train_bpe(
    input_path: str | os.PathLike,
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
            f, desired_num_chunks=kwargs.get("desired_num_chunks", NUM_PROCESSES), split_special_token=b"\n"
        )

    # 1.2 Count word frequencies across chunks using multiprocessing
    manager = Manager()
    queue = manager.Queue()
    processes: list[Process] = []

    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string_worker,
            args=(input_path, special_tokens, queue, start, end, False),
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
    pairs_freqs = pair_counts(word_counter)
    for _ in range(num_merges):
        most_frequent_pair = get_most_frequent_pair(pairs_freqs, vocab)
        new_id = add_pair_to_vocab(vocab, most_frequent_pair)
        word_counter, pairs_freqs = merge_pairs(word_counter, most_frequent_pair, new_id)

        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

    return vocab, merges
