import json
import os
import time
from functools import wraps
from typing import BinaryIO


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


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIME] {func.__name__} took {end - start:.2f}s")

        return result

    return wrapper
