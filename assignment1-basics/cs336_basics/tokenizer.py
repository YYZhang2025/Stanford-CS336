import json
import os
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Manager, Process, Queue
from queue import Empty
from typing import BinaryIO

import regex as re
from tqdm import trange

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
    if not special_tokens:
        return [text]

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
) -> tuple[Counter, dict[tuple[int, int], int]]:
    word_counter = Counter()
    pairs: dict[tuple[int, int], int] = {}

    chunks = split_by_special_tokens(string, special_tokens, include_special=including_special)

    for chunk in chunks:
        if including_special and chunk in special_tokens:
            word_counter[tuple(string_to_bytes(chunk))] += 1
        else:
            for match in re.finditer(PAT, chunk):
                word = match.group(0)
                word_encoded = tuple(string_to_bytes(word, return_int=True))
                word_counter[word_encoded] += 1

    for word in word_counter:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + word_counter[word]

    return word_counter, pairs


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

    word_counter, pairs_counter = pre_tokenize(chunk, special_tokens, include_special)

    # Put the result in the queue
    queue.put((word_counter, pairs_counter))


### --------- End Pre-process steps --------------
# def pair_counts(
#     word_counter: dict[tuple[int, ...], int],
# ) -> dict[tuple[int, int], int]:
#     pairs: dict[tuple[int, int], int] = {}

#     for token, freq in word_counter.items():
#         for i in range(len(token) - 1):
#             pair = (token[i], token[i + 1])
#             pairs[pair] = pairs.get(pair, 0) + freq

#     return pairs


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


import heapq


def build_pair_heap(pair_counter: Counter, vocab: dict[int, bytes]):
    heap = []
    for pair, freq in pair_counter.items():
        if freq > 0:
            heapq.heappush(heap, (-freq, (vocab[pair[0]], vocab[pair[1]]), pair))
    return heap


def pop_best_pair(heap, pair_counter: Counter, vocab: dict[int, bytes]) -> tuple[int, int]:
    # Lazy deletion: discard stale heap entries until top matches current counter & vocab tie-break key.
    while True:
        neg_f, vocab_a, vocab_b, pair = heap[0]
        cur_f = pair_counter.get(pair, 0)
        if cur_f <= 0:
            heapq.heappop(heap)
            continue
        if -neg_f != cur_f:
            heapq.heappop(heap)
            continue
        a, b = pair
        if vocab_a != vocab[a] or vocab_b != vocab[b]:
            heapq.heappop(heap)
            continue
        return pair


def merge_pairs(
    word_counter: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, int], int]]:
    new_word_counter: defaultdict[tuple[int, ...], int] = defaultdict(int)
    updated_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    a, b = pair
    for word, freq in word_counter.items():
        new_word = []
        i = 0
        L = len(word)
        new_word_append = new_word.append

        while i < L:
            if i + 1 < L and word[i] == a and word[i + 1] == b:
                new_word_append(new_id)
                i += 2
            else:
                new_word_append(word[i])
                i += 1

        new_word_counter[tuple(new_word)] += freq

        if len(new_word) >= 2:
            prev = new_word[0]
            for cur in new_word[1:]:
                updated_pair_counts[(prev, cur)] += freq
                prev = cur

    return new_word_counter, updated_pair_counts


def merge_pairs_incremental(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int, ...], int], Counter]:
    a, b = pair
    new_word_counter: defaultdict[tuple[int, ...], int] = defaultdict(int)
    updated_pair_counter: Counter = pair_counter.copy()

    for word, freq in word_counter.items():
        w = word
        L = len(w)

        # Fast path: check if `pair` occurs; if not, keep the word and skip updates.
        i = 0
        found = False
        while i + 1 < L:
            if w[i] == a and w[i + 1] == b:
                found = True
                break
            i += 1

        if not found:
            new_word_counter[w] += freq
            continue

        # (1) subtract old adjacent pairs for this word
        if L >= 2:
            prev = w[0]
            for cur in w[1:]:
                updated_pair_counter[(prev, cur)] -= freq
                prev = cur

        # (2) build merged word
        out: list[int] = []
        out_append = out.append
        i = 0
        while i < L:
            if i + 1 < L and w[i] == a and w[i + 1] == b:
                out_append(new_id)
                i += 2
            else:
                out_append(w[i])
                i += 1
        new_word_counter[tuple(out)] += freq

        # (3) add new adjacent pairs for merged word
        if len(out) >= 2:
            prev = out[0]
            for cur in out[1:]:
                updated_pair_counter[(prev, cur)] += freq
                prev = cur

    for k in list(updated_pair_counter.keys()):
        if updated_pair_counter[k] <= 0:
            del updated_pair_counter[k]

    return new_word_counter, updated_pair_counter


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_dir: str | os.PathLike,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vocab_filepath = os.path.join(output_dir, "vocab.json")
    merges_filepath = os.path.join(output_dir, "merges.txt")

    # Save vocab
    vocab_inv = {v.decode("latin1"): k for k, v in vocab.items()}
    with open(vocab_filepath, "w") as vf:
        json.dump(vocab_inv, vf, ensure_ascii=False, indent=2)

    # Save merges
    with open(merges_filepath, "w") as mf:
        mf.write("#version: 0.2\n")
        for a, b in merges:
            mf.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # start_time = time.time()
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 1. Pre-tokenization
    # 1.1 Find chunk boundaries
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, desired_num_chunks=kwargs.get("desired_num_chunks", NUM_PROCESSES), split_special_token=b"\n"
        )

    if verbose:
        print(f"Identified {len(chunk_boundaries) - 1} chunks for pre-tokenization.")

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

    if verbose:
        print("Pre-tokenization processes completed. Aggregating results...")

    word_counter = Counter()
    pairs_freqs = Counter()
    for _ in range(len(processes)):
        try:
            partial_counter, partial_pairs = queue.get(timeout=10)
            word_counter.update(partial_counter)
            pairs_freqs.update(partial_pairs)
        except Empty:
            continue
    if verbose:
        print(f"Completed pre-tokenization. Vocabulary size: {len(word_counter)} unique tokens.")
    # 2. BPE Core Loop

    for _ in trange(num_merges):
        most_frequent_pair = get_most_frequent_pair(pairs_freqs, vocab)
        new_id = add_pair_to_vocab(vocab, most_frequent_pair)
        # word_counter, pairs_freqs = merge_pairs(word_counter, most_frequent_pair, new_id)
        word_counter, pairs_freqs = merge_pairs_incremental(
            word_counter, pairs_freqs, most_frequent_pair, new_id
        )

        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

    # end_time = time.time()
    # print(f"BPE training completed in {end_time - start_time:.2f} seconds.")
    if kwargs.get("save_path"):
        save_vocab_and_merges(vocab, merges, kwargs["save_path"])

    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [t.encode("utf-8") for t in self.special_tokens]

        self.vocab_inv = {v: k for k, v in self.vocab.items()}

    def _pre_tokenize(self, text: str) -> list[bytes]:
        """Pre-tokenize the input text into a list of byte-strings.

        Returns a list where each element is:
          - the UTF-8 bytes of a special token (e.g. b"<|endoftext|>")
          - the UTF-8 bytes of a regex token (e.g. b" hello")
        """
        parts = split_by_special_tokens(text, self.special_tokens, include_special=True)
        token_list: list[bytes] = []

        for part in parts:
            if part == "":
                continue
            if part in self.special_tokens:
                token_list.append(part.encode("utf-8"))
            else:
                for tok in re.findall(PAT, part):
                    # Each regex token becomes a single bytestring.
                    token_list.append(tok.encode("utf-8"))

        return token_list

    def encode(self, text: str) -> list[int]:
        byte_tokens = self._pre_tokenize(text)

        # Precompute merge -> new_id once (skip merges that don't exist in vocab_inv)
        merge_to_new_id: dict[tuple[bytes, bytes], int] = {}
        for a, b in self.merges:
            new_id = self.vocab_inv.get(a + b)
            if new_id is not None:
                merge_to_new_id[(a, b)] = new_id

        # Convert byte tokens to initial ids (byte-level)
        token_ids: list[list[int]] = []
        special_set = set(self.special_tokens_bytes)
        for btok in byte_tokens:
            if btok in special_set:
                token_ids.append([self.vocab_inv[btok]])
            else:
                # btok is a bytestring; iterating yields ints, so convert to single-byte bytes keys.
                token_ids.append([self.vocab_inv[bytes([b])] for b in btok])

        # Apply merges in learned order
        for i, pretoken in enumerate(token_ids):
            for a, b in self.merges:
                new_index = merge_to_new_id.get((a, b))
                if new_index is None:
                    continue

                merged: list[int] = []
                j = 0
                L = len(pretoken)
                while j < L:
                    if j + 1 < L and (self.vocab[pretoken[j]], self.vocab[pretoken[j + 1]]) == (a, b):
                        merged.append(new_index)
                        j += 2
                    else:
                        merged.append(pretoken[j])
                        j += 1
                pretoken = merged

            token_ids[i] = pretoken

        return [idx for pre in token_ids for idx in pre]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Placeholder for iterable encoding logic
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character

        tokens = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8", errors="replace")

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        with open(vocab_filepath, "r") as vf:
            vocab_data = json.load(vf)
            vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

        merges = []
        with open(merges_filepath, "r") as mf:
            for line in mf:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append((bytes(parts[0], "latin1"), bytes(parts[1], "latin1")))

        return cls(vocab, merges, special_tokens)
