from typing import Dict, Tuple, List, Iterable, Iterator
from collections import Counter
import regex as re

from queue import Empty
from multiprocessing import Process, Queue, Manager


from tqdm import trange, tqdm
import pickle

import os 


from cs336_basics.tokenizer.pretokenization_regular_pattern import PAT
from cs336_basics.tokenizer.utils import find_chunk_boundaries 


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
    text: str, special_tokens: list[str]
) -> List[str]:
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    if not special_tokens_sorted:
        return [text]
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)
    special_chunks = re.split(f"({pattern})", text)

    return special_chunks



def pre_tokenize_string(text: str, special_tokens: List[str], include_special: bool = False) -> Counter:
    word_counter = Counter()
    special_chunks = split_by_special_tokens(text, special_tokens)

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



class Tokenizer:
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = []
    ):
        self.vocab = vocab
        self.merges = merges
        
        # self.register_special_tokens(special_tokens)
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        
        if special_tokens is None:
            self.special_tokens = {}
            self.bytes_special_tokens = []
        else:
            self.special_tokens = {token: i for i, token in enumerate(special_tokens, start=len(self.vocab))}
            self.bytes_special_tokens = [token.encode("utf-8") for token in special_tokens if isinstance(token, str)]

        
        
    
    def register_special_tokens(self, special_tokens):
        if special_tokens is None:
            self.special_tokens = {}
            self.bytes_special_tokens = []
            return
        
        if not all(isinstance(token, bytes) for token in special_tokens):
            bytes_special_tokens = [token.encode("utf-8") for token in special_tokens if isinstance(token, str)]
            
        for i, token in enumerate(bytes_special_tokens, start=len(self.vocab)):
            # Add special tokens to the vocabulary
            self.vocab[i] = token
            
        # self.bytes_special_tokens = bytes_special_tokens
        # self.special_tokens = {token: i for i, token in enumerate(special_tokens, start=len(self.vocab))}
        
    def _pre_tokenize(self, text) -> List[bytes]:
        """
        Pre-tokenize the input text into bytes.
        """
        parts = split_by_special_tokens(text, list(self.special_tokens.keys()))
        token_list = []
        
        for part in parts:
            if part in self.special_tokens.keys():
                token_list.append(part.encode("utf-8"))
            else:
                tokens = re.findall(PAT, part)
                token_list.extend(word_to_bytes(token) for token in tokens)

        return token_list
    

    
    def encode(self, text: str) -> List[int]:
        byte_tokens = self._pre_tokenize(text)
        

        # Convert byte tokens to indices
        token_ids = []
        for byte_token in byte_tokens:
            # print(f"Processing byte token: {byte_token}")
            if byte_token in self.bytes_special_tokens:
                token_ids.append([self.vocab_inv[byte_token]])
            else:
                token_ids.append([self.vocab_inv[b] for b in byte_token]) #type: ignore

        for i, pretoken in enumerate(token_ids):
            for merge in self.merges:
                new_index = self.vocab_inv.get(merge[0] + merge[1], None)
                if new_index is None:
                    continue

                merged = []
                j = 0
                while j < len(pretoken):
                    if (
                        j < len(pretoken) - 1
                        and (self.vocab[pretoken[j]], self.vocab[pretoken[j + 1]]) == merge
                    ):
                        merged.append(new_index)
                        j += 2
                    else:
                        merged.append(pretoken[j])
                        j += 1
                        
                pretoken = merged
            token_ids[i] = pretoken[:]

        return [i for pre in token_ids for i in pre]
        

    

    def encode_iterable(self, iterable: Iterable[str], batch_size: int = 1024) -> Iterator[int]:
        """
        Encode lines of text from an iterable using buffered batching.
        This version preserves newlines by assuming the input was split with `splitlines(keepends=True)`.
        """
        batch = []
        for line in tqdm(iterable):
            if not line:
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                for encoded in map(self.encode, batch):
                    yield from encoded
                batch.clear()
                
        if batch:
            for encoded in map(self.encode, batch):
                yield from encoded
    
    def decode(self, ids: list[int]) -> str:
        # https://en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character
        
        tokens = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)
        return tokens.decode("utf-8", errors="replace")
    
    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ):
        with open(vocab_path, 'rb') as vf:
            raw_vocab = pickle.load(vf)

        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in raw_vocab.items()}

        with open(merges_path, 'rb') as mf:
            raw_merges = pickle.load(mf)

        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)