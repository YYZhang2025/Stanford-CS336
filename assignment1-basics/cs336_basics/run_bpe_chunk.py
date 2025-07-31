from typing import List, Tuple, Dict
from dataclasses import dataclass
from typing import Dict, Tuple, List,  BinaryIO
import regex as re
from multiprocessing import Process, Queue

from collections import Counter

import os 



PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
FILE_PATH = "../data/TinyStoriesV2-GPT4-valid.txt"

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

def word_to_bytes(word: str, num_special_tokens: int = 1) -> bytes:
    """
    Convert a word to bytes.
    """
    byte_ids = [b + num_special_tokens for b in word.encode('utf-8')]
    return bytes(byte_ids)



def pre_tokenize_string(chunk: str, special_tokens: List[str], drop_special_token: bool = True) -> Dict[Tuple[bytes], int]:
    # 1. Split the chunk into words using regex
    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    tokens = []
    if not special_tokens_sorted:
        return [chunk]
    else:
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        tokens = re.split(f"({pattern})", chunk)
        if drop_special_token:
            tokens = [tok for tok in tokens if tok not in special_tokens_sorted]

    # print(f"Tokens found: {tokens}")
    print(f"[子进程] PID: {os.getpid()}")
    
    # 2. Convert each word to bytes
    word_byte_counter = Counter()
    for word in tokens:
        if word:
            byte_word = word_to_bytes(word, num_special_tokens=len(special_tokens))
            word_byte_counter[tuple(byte_word)] += 1    
    return word_byte_counter

    


def process_chunk(path: str, start: int, end: int,  q: Queue, **kwargs):
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        word_byte_counter = pre_tokenize_string(chunk, special_tokens=kwargs.get('special_tokens', []), drop_special_token=kwargs.get('drop_special_token', True))
        
        print(f"[子进程] PID: {os.getpid()}, Chunk size: {len(chunk)} bytes, Word count: {len(word_byte_counter)}"  )
        q.put(word_byte_counter)
        
        print(f"[子进程] PID: {os.getpid()} put() success: {len(word_byte_counter)} items")
        
        
# Encoder special tokens from string to bytes
def encode_special_tokens(special_tokens: list[str]) -> list[bytes]:
    """
    Convert a list of special tokens from strings to bytes.
    """
    return [token.encode('utf-8') for token in special_tokens]



def main():
    FILE_PATH = "./data/TinyStoriesV2-GPT4-valid.txt"
    
    print(os.getcwd())
    special_tokens = ['<|endoftext|>']
    num_processes = 4
    special_tokens_bytes = encode_special_tokens(['<|endoftext|>'])

    boundaries = find_chunk_boundaries(open(FILE_PATH, "rb"), num_processes, b'<|endoftext|>')

    processes = []
    from multiprocessing import Manager

    manager = Manager()
    q = manager.Queue()
    print(f"Chunk boundaries: {boundaries}")
    print(f"Number of processes: {num_processes}")
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        p = Process(target=process_chunk, args=(FILE_PATH, start, end), kwargs={'q': q, 'special_tokens': special_tokens})
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
        
    print("All processes finished.")
    from queue import Empty

    results = []
    for _ in range(len(processes)):
        try:
            results.append(q.get(timeout=10))  # 最多等待 10 秒
        except Empty:
            print("⚠️ 警告：某个子进程没有返回结果！")
    
    # Combine results
    combined_counter = Counter()
    for result in results:
        combined_counter.update(result)
    
    # print(combined_counter)s
    print("Tokenization complete. Program finished.")
    print(f"Total unique tokens: {len(combined_counter)}")
    
    print("Sample of token counts:")
    for token, count in list(combined_counter.items())[:10]:  # Print first 10 tokens
        print(f"Token: {token}, Count: {count}")    
    


if __name__ == "__main__":
    main()
    