from typing import List, Tuple, Dict
from dataclasses import dataclass
from typing import Dict, Tuple, List,  BinaryIO
import regex as re
from queue import Empty
from multiprocessing import Process, Queue, Manager

from collections import Counter

import os 


from cs336_basics.pretokenization_regular_pattern import PAT
from cs336_basics.utils import find_chunk_boundaries 


def initialize_vocab(special_tokens: List[bytes]) -> Dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}  # ASCII characters
    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token

    return vocab

def word_to_bytes(word: str) -> List[bytes]:
    """
    Convert a word to bytes.
    """
    byte_ids = [bytes([b]) for b in word.encode("utf-8")]

    return byte_ids


def split_by_special_tokens(
    text: str, special_tokens: list[str], include_special: bool = False
) -> List[str]:
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if include_special:
        special_chunks = re.split(f"({pattern})", text)
    else:
        # Split without capturing the special tokens
        special_chunks = re.split(pattern, text)

    return special_chunks

# def pre_tokenize_string(chunk: str, special_tokens: List[bytes], drop_special_token: bool = True) -> Dict[Tuple[bytes], int]:
#     # 1. Split the chunk into words using regex
#     special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
#     tokens = []
#     if not special_tokens_sorted:
#         return [chunk]
#     else:
#         pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
#         tokens = re.split(f"({pattern})", chunk)
#         if drop_special_token:
#             tokens = [tok for tok in tokens if tok not in special_tokens_sorted]

#     # 2. Convert each word to bytes
#     word_byte_counter = Counter()
#     for word in tokens:
#         if word:
#             byte_word = word_to_bytes(word, num_special_tokens=len(special_tokens))
#             word_byte_counter[tuple(byte_word)] += 1    
#     return word_byte_counter

def pre_tokenize_string(
    input_path: str | os.PathLike, special_tokens: list[str], queue: Queue, start: int, end: int, include_special: bool = False,
):
    """
    Pre-tokenize a string into bytes.
    """

    word_counter = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    special_chunks = split_by_special_tokens(chunk, special_tokens, include_special)

    for chunk in special_chunks:
        if chunk in special_tokens:
            if include_special:
                token = tuple(word_to_bytes(chunk))
                word_counter[token] += 1
        else:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                token = tuple(word_to_bytes(word))
                word_counter[token] += 1
                
    # Put the result in the queue
    queue.put(word_counter)

    # return word_counter


def pair_counts(
    word_counter: Dict[Tuple[bytes], int],
) -> Dict[Tuple[bytes, bytes], int]:
    """
    Count pairs of bytes in the word counter.
    """
    pairs: Dict[Tuple[bytes, bytes], int] = {}
    for token, freq in word_counter.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pairs[pair] = pairs.get(pair, 0) + freq

    return pairs


def get_most_frequent_pair(
    pairs: Dict[Tuple[bytes, bytes], int],
) -> Tuple[bytes, bytes]:
    max_freq = max(pairs.values())
    candidates = [pair for pair, freq in pairs.items() if freq == max_freq]
    res = max(candidates)

    return res


def add_pair_to_vocab(
    vocab: Dict[int, bytes], pair: Tuple[bytes, bytes], vocab_inv: Dict[bytes, int]
) -> int:
    """
    Add a new pair to the vocabulary.
    """
    index = len(vocab)
    s = vocab[vocab_inv[pair[0]]] + vocab[vocab_inv[pair[1]]]
    vocab[index] = s
    vocab_inv[vocab[index]] = index

    return index

from collections import Counter, defaultdict


def merge_pair(
    word_counter: Dict[Tuple[bytes], int], pair: Tuple[bytes, bytes]
) -> Tuple[Dict[Tuple[bytes], int], Dict]:
    """
    Merge a pair of bytes in the word counter.
    """
    new_word_counter = Counter()
    updated_pair_counts = defaultdict(int)

    for token, freq in word_counter.items():
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                new_token.append(token[i] + token[i + 1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_word_counter[tuple(new_token)] += freq

        for j in range(len(new_token) - 1):
            new_pair = (new_token[j], new_token[j + 1])
            updated_pair_counts[new_pair] += freq

    return new_word_counter, updated_pair_counts


def check_and_convert_special_tokens(
    special_tokens: List[str] | List[bytes],
) -> List[bytes]:
    """
    Check if special tokens are in the vocabulary and convert them to bytes.
    """
    if not all(isinstance(token, bytes) for token in special_tokens):
        special_tokens_bytes = [
            token.encode("utf-8") for token in special_tokens if isinstance(token, str)
        ]

    return special_tokens_bytes


def train_bpe(
    input_path: str | os.PathLike ,
    vocab_size=10_000,
    special_tokens: List[str] = [],
    **kwargs,
):
    special_tokens_bytes = check_and_convert_special_tokens(special_tokens)

    vocab = initialize_vocab(special_tokens_bytes)
    vocab_inv = {v: k for k, v in vocab.items()}
    merges: List[Tuple[bytes, bytes]] = []
    
    
    # Pre-tokenization
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, kwargs.get("num_processes", 8), special_tokens_bytes[0]
        )

    
    manager = Manager()
    queue = manager.Queue()
    processes = []
    
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string,
            args=(input_path, special_tokens, queue, start, end, False),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    word_counter = Counter()
    for _ in range(len(processes)):
        try:
            word_counter += queue.get(timeout=10)  # Wait up to 10 seconds for results
        except Empty:
            print("⚠️ Warning: A subprocess did not return a result!")

    # End Pre-tokenization

    pairs_freqs = pair_counts(word_counter)

    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):

        most_common_pair = get_most_frequent_pair(pairs_freqs)

        new_index = add_pair_to_vocab(vocab, most_common_pair, vocab_inv)
        merges.append(most_common_pair)

        word_counter, pairs_freqs = merge_pair(word_counter, most_common_pair)

    return vocab, merges
