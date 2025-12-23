import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Manager, Process, Queue
from queue import Empty

import regex as re
from tqdm import trange

from cs336_basics.tokenizer.merge_fn import (
    build_pair_heap,
    get_most_frequent_pair,
    merge_pairs,
    merge_pairs_incremental,
    merge_pairs_with_heap,
    merge_pairs_with_heap_index,
    pop_best_pair,
)
from cs336_basics.tokenizer.utils import (
    find_chunk_boundaries,
    save_vocab_and_merges,
    string_to_bytes,
    timeit,
    utf8_bytes_to_string,
)
from cs336_basics.utils import print_color

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = max(1, (os.cpu_count() or 1) - 4)


def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # idx -> byte representation
    current_index = 256

    if special_tokens is not None:
        for token in special_tokens:
            vocab[current_index] = token.encode("utf-8")
            current_index += 1

    return vocab


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


@timeit
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    # 1. Pre-tokenization
    # 1.1 Find chunk boundaries
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, desired_num_chunks=kwargs.get("desired_num_chunks", NUM_PROCESSES), split_special_token=b"\n"
        )

    if verbose:
        print_color(f"Identified {len(chunk_boundaries) - 1} chunks for pre-tokenization.")

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
        print_color("Pre-tokenization processes completed. Aggregating results...")

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
        print_color(f"Completed pre-tokenization. Vocabulary size: {len(word_counter)} unique tokens.")

    for word in word_counter:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_to_words[pair].add(word)

    # 2. BPE Core Loop
    pair_heap = build_pair_heap(pairs_freqs, vocab)

    for i in trange(num_merges):
        most_frequent_pair = pop_best_pair(pair_heap, pairs_freqs, vocab)
        # most_frequent_pair = get_most_frequent_pair(pairs_freqs, vocab)

        new_id = len(vocab)
        vocab[new_id] = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]

        # word_counter, pairs_freqs, pair_heap = merge_pairs_with_heap(
        #     word_counter, pairs_freqs, most_frequent_pair, new_id, vocab, pair_heap
        # )
        # word_counter, pairs_freqs = merge_pairs_incremental(
        #     word_counter, pairs_freqs, most_frequent_pair, new_id
        # )

        word_counter, pairs_freqs, pair_heap, pair_to_words = merge_pairs_with_heap_index(
            word_counter, pairs_freqs, most_frequent_pair, new_id, vocab, pair_heap, pair_to_words
        )
        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

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
        self.special_set = set(self.special_tokens_bytes)

        self.vocab_inv = {v: k for k, v in self.vocab.items()}

        rank: dict[tuple[int, int], int] = {}
        for r, (a_bytes, b_bytes) in enumerate(self.merges):
            a_id = self.vocab_inv.get(a_bytes)
            b_id = self.vocab_inv.get(b_bytes)
            if a_id is None or b_id is None:
                continue
            rank[(a_id, b_id)] = r
        self.rank = rank

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

        def merge_one_pretoken(ids: list[int]) -> list[int]:
            n = len(ids)
            if n <= 1:
                return ids

            # Doubly-linked list over positions 0..n-1 (positions are stable; nodes get "deleted")
            prev = [-1] * n
            nxt = [-1] * n
            for i in range(n):
                prev[i] = i - 1
                nxt[i] = i + 1 if i + 1 < n else -1

            alive = [True] * n

            # best pair per left-position i: (rank, i)

            heap: list[tuple[int, int]] = []

            def pair_rank(i: int) -> int | None:
                j = nxt[i]
                if j == -1 or not alive[i] or not alive[j]:
                    return None
                return self.rank.get((ids[i], ids[j]))

            def push_if_valid(i: int):
                r = pair_rank(i)
                if r is not None:
                    heapq.heappush(heap, (r, i))

            for i in range(n):
                push_if_valid(i)

            # We need to create new ids when merging. We can use vocab_inv on concatenated bytes:
            # new_id = vocab_inv[vocab[a_id] + vocab[b_id]]
            # (this should exist because training added these merges to vocab)
            while heap:
                r, i = heapq.heappop(heap)
                j = nxt[i]
                if j == -1 or not alive[i] or not alive[j]:
                    continue
                # stale check: rank might no longer match current neighbor
                cur_r = self.rank.get((ids[i], ids[j]))
                if cur_r is None or cur_r != r:
                    continue

                # merge i and j into i
                new_bytes = self.vocab[ids[i]] + self.vocab[ids[j]]
                new_id = self.vocab_inv[new_bytes]
                ids[i] = new_id

                # delete j from the linked list
                alive[j] = False
                nj = nxt[j]
                nxt[i] = nj
                if nj != -1:
                    prev[nj] = i

                # Only pairs that can change are around i (prev[i], i) and (i, nxt[i])
                pi = prev[i]
                if pi != -1:
                    push_if_valid(pi)
                push_if_valid(i)

            # materialize result by walking the linked list
            out: list[int] = []
            k = 0
            while k != -1:
                if alive[k]:
                    out.append(ids[k])
                k = nxt[k]
            return out

        token_ids: list[int] = []
        for btok in byte_tokens:
            if btok in self.special_set:
                token_ids.append(self.vocab_inv[btok])
            else:
                ids = [self.vocab_inv[bytes([b])] for b in btok]
                token_ids.extend(merge_one_pretoken(ids))

        return token_ids

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
