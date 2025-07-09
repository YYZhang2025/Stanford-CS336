import os
import regex as re
from typing import List, Tuple, BinaryIO, Iterable, Iterator
from collections import defaultdict
from multiprocessing import Process, Queue

# === 正则分词模式 ===
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# === 文本切分：保留特殊 token ===
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
            tokens = re.findall(PATTERN, part)
            tokens_list.extend(token.encode("utf-8") for token in tokens)
    return tokens_list


# === 并行分词 worker ===
def worker(text: str, special_tokens: list[str], q: Queue):
    q.put(pretokenize(text, special_tokens))


# === 按特殊 token 查找 chunk 边界 ===
def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_pos = chunk_boundaries[bi]
        file.seek(initial_pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_pos + found_at
                break
            initial_pos += mini_chunk_size

    return sorted(set(chunk_boundaries))


# === 合并最频繁的 token 对 ===
def merge(
    counts: dict[Tuple[int, int], int],
    index_dict: dict[Tuple[int, int], set[int]],
    pretokens: list[list[int]],
    max_pair: Tuple[int, int],
    new_index: int,
):
    affected_indices = index_dict[max_pair]
    for i in affected_indices:
        pretoken = pretokens[i]
        new_pretoken = []
        merge_positions = []

        j = 0
        pos = 0
        while j < len(pretoken):
            if j < len(pretoken) - 1 and (pretoken[j], pretoken[j + 1]) == max_pair:
                new_pretoken.append(new_index)
                merge_positions.append(pos)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
            pos += 1

        for pos in merge_positions:
            counts[max_pair] -= 1
            if pos > 0:
                prev = new_pretoken[pos - 1]
                old_left = (
                    (prev, max_pair[0])
                    if prev != new_index
                    else (max_pair[1], max_pair[0])
                )
                counts[old_left] -= 1
                new_left = (prev, new_index)
                counts[new_left] += 1
                index_dict[new_left].add(i)
            if pos < len(new_pretoken) - 1:
                next_tok = new_pretoken[pos + 1]
                old_right = (
                    (max_pair[1], next_tok)
                    if next_tok != new_index
                    else (max_pair[1], max_pair[0])
                )
                counts[old_right] -= 1
                new_right = (new_index, next_tok)
                counts[new_right] += 1
                index_dict[new_right].add(i)

        pretokens[i] = new_pretoken


# === BPE 训练主函数 ===
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | dict[str, int],
    **kwargs,
) -> Tuple[dict[int, bytes], list[Tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    vocab = {x: bytes([x]) for x in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    merges = []

    num_processes = kwargs.get("num_processes", os.cpu_count() or 4)
    chunk_list = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries, boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    q = Queue()
    processes = [
        Process(target=worker, args=(chunk, special_tokens, q)) for chunk in chunk_list
    ]
    for p in processes:
        p.start()
    pretokens_list = [q.get() for _ in processes]
    for p in processes:
        p.join()

    # Convert to list[list[int]]
    pretokens = []
    for token_bytes in [token for tokens in pretokens_list for token in tokens]:
        pretokens.append([b for b in token_bytes])

    counts = defaultdict(int)
    index_dict = defaultdict(set)
    for j, pretoken in enumerate(pretokens):
        for a, b in zip(pretoken, pretoken[1:]):
            counts[(a, b)] += 1
            index_dict[(a, b)].add(j)

    for i in range(num_merges):
        if not counts:
            break
        max_pair = max(
            counts.items(),
            key=lambda x: (
                x[1],
                vocab[x[0][0]].decode("utf-8", errors="ignore"),
                vocab[x[0][1]].decode("utf-8", errors="ignore"),
            ),
        )[0]
        new_index = 256 + len(special_tokens) + i
        vocab[new_index] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
        merge(counts, index_dict, pretokens, max_pair, new_index)

    return vocab, merges


class BPETokenizer:
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
