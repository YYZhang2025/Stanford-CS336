from typing import List, Tuple, Dict
from typing import Dict, Tuple, List, Iterable, Iterator
import regex as re
from queue import Empty
from multiprocessing import Process, Queue, Manager

from collections import Counter

from tqdm import trange

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


def pre_tokenize_string(text: str, special_tokens: List[str], include_special: bool = False) -> Counter:
    word_counter = Counter()
    special_chunks = split_by_special_tokens(text, special_tokens, include_special)

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

    return word_counter


# TODO: Implement the worker for this.
def pre_tokenize_string_worker(
    input_path: str | os.PathLike, special_tokens: list[str], queue: Queue, start: int, end: int, include_special: bool = False,
):
    """
    Pre-tokenize a string into bytes.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
    word_counter = pre_tokenize_string(chunk, special_tokens, include_special)
                
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
            word_counter += queue.get(timeout=10)  # Wait up to 10 seconds for results
        except Empty:
            print("⚠️ Warning: A subprocess did not return a result!")

    # End Pre-tokenization

    pairs_freqs = pair_counts(word_counter)
    
    num_merges = vocab_size - len(vocab)
    for _ in trange(num_merges):

        most_common_pair = get_most_frequent_pair(pairs_freqs)

        new_index = add_pair_to_vocab(vocab, most_common_pair, vocab_inv)
        merges.append(most_common_pair)

        word_counter, pairs_freqs = merge_pair(word_counter, most_common_pair)

    return vocab, merges

def split_by_special_tokens(text: str, special_tokens: list[str]) -> List[str]:
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    if not special_tokens_sorted:
        return [text]
    pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
    return re.split(f"({pattern})", text)


# === 预分词 ===
def pretokenize(
    text: str, special_tokens: list[str], drop_special_token: bool = True
) -> List[bytes]:
    parts = split_by_special_tokens(text, special_tokens)
    tokens_list = []
    for part in parts:
        if part in special_tokens:
            if not drop_special_token:
                tokens_list.append(part.encode("utf-8"))
        else:
            tokens = re.findall(PAT, part)
            tokens_list.extend(token.encode("utf-8") for token in tokens)
            
    return tokens_list



class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    def encode(self, text: str) -> list[int]:
        vocab_rev = {v: k for k, v in self.vocab.items()}
        byte_tokens = pretokenize(text, self.special_tokens, drop_special_token=False)
        pretokens = []
        for bt in byte_tokens:
            if bt in [tok.encode() for tok in self.special_tokens]:
                pretokens.append([vocab_rev[bt]])
            else:
                pretokens.append([vocab_rev[bytes([b])] for b in bt])

        for i, pretoken in enumerate(pretokens):
            for merge in self.merges:
                new_index = vocab_rev[merge[0] + merge[1]]
                merged = []
                j = 0
                while j < len(pretoken):
                    if (
                        j < len(pretoken) - 1
                        and (self.vocab[pretoken[j]], self.vocab[pretoken[j + 1]])
                        == merge
                    ):
                        merged.append(new_index)
                        j += 2
                    else:
                        merged.append(pretoken[j])
                        j += 1
                pretoken = merged
            pretokens[i] = pretoken

        return [id for pre in pretokens for id in pre]

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ):
        raise NotImplementedError
