import heapq
from collections import Counter, defaultdict


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


# Version 1: Simple pair merging without efficient updates
def merge_pairs(
    word_counter: dict[tuple[int, ...], int],
    target_pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, int], int]]:
    new_word_counter: defaultdict[tuple[int, ...], int] = defaultdict(int)
    updated_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    a, b = target_pair
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


# Version 2: Incremental pair merging with efficient updates
def merge_pairs_incremental(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    target_pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int, ...], int], Counter]:
    a, b = target_pair
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


# Version 3: Using heap for pair selection


# _INV = bytes.maketrans(bytes(range(256)), bytes([255 - i for i in range(256)]))
# def inv_bytes(b: bytes) -> bytes:
#     return b.translate(_INV)


class HeapItem:
    def __init__(self, neg_freq: int, pair_bytes: tuple[bytes, bytes], pair: tuple[int, int]):
        self.neg_freq = neg_freq
        self.pair_bytes = pair_bytes
        self.pair = pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        return self.pair_bytes > other.pair_bytes  # reverse order for max-heap behavior


def build_pair_heap(pairs_freqs: Counter, vocab: dict[int, bytes]):
    heap = []
    for (a, b), f in pairs_freqs.items():
        if f > 0:
            item = HeapItem(-f, (vocab[a], vocab[b]), (a, b))
            heapq.heappush(heap, item)
            # heapq.heappush(heap, (-f, (inv_bytes(vocab[a]), inv_bytes(vocab[b])), (a, b)))
    return heap


def pop_best_pair(heap, pairs_freqs: Counter, vocab: dict[int, bytes]) -> tuple[int, int]:
    while heap:
        # neg_f, pair_vocab, pair = heap[0]
        item = heap[0]
        neg_f = item.neg_freq
        pair = item.pair
        cur_f = pairs_freqs.get(pair, 0)
        if cur_f <= 0 or -neg_f != cur_f:  # frequency changed, which means the pair we store in heap is stale
            heapq.heappop(heap)
            continue

        return pair
    raise ValueError("No positive-frequency pairs remain")


def merge_pairs_with_heap(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    target_pair: tuple[int, int],
    new_id: int,
    vocab: dict[int, bytes],
    pair_heap,
) -> tuple[dict[tuple[int, ...], int], Counter, list]:
    a, b = target_pair
    new_word_counter: defaultdict[tuple[int, ...], int] = defaultdict(int)
    updated_pair_counter: Counter = pair_counter.copy()
    changed_pairs = set()

    # For each word, perform the merge and update pair counts incrementally
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
                changed_pairs.add((prev, cur))
                prev = cur

        # (2) build merged word
        out: list[int] = []
        i = 0
        while i < L:
            if i + 1 < L and w[i] == a and w[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(w[i])
                i += 1
        new_word_counter[tuple(out)] += freq

        # (3) add new adjacent pairs for merged word
        if len(out) >= 2:
            prev = out[0]
            for cur in out[1:]:
                updated_pair_counter[(prev, cur)] += freq
                changed_pairs.add((prev, cur))
                prev = cur

    for k in list(updated_pair_counter.keys()):
        if updated_pair_counter[k] <= 0:
            del updated_pair_counter[k]

    # Update the heap with new pair frequencies
    if pair_heap is not None:
        for pair in changed_pairs:
            freq = updated_pair_counter.get(pair, 0)
            # heapq.heappush(pair_heap, (-freq, (inv_bytes(vocab[pair[0]]), inv_bytes(vocab[pair[1]])), pair))
            heapq.heappush(pair_heap, HeapItem(-freq, (vocab[pair[0]], vocab[pair[1]]), pair))

    return new_word_counter, updated_pair_counter, pair_heap


# Version 4:


def merge_pairs_with_heap_index(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    target_pair: tuple[int, int],
    new_id: int,
    vocab: dict[int, bytes],
    pair_heap,
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> tuple[
    dict[tuple[int, ...], int],
    Counter,
    list,
    dict[tuple[int, int], set[tuple[int, ...]]],
]:
    """
    Merge `target_pair=(a,b)` into token `new_id` across the corpus, updating:
      - word_counter (counts of tokenized "words" as tuples of ids)
      - pair_counter (counts of adjacent pairs across all words)
      - pair_to_words (index: pair -> set of words that contain that pair)
      - pair_heap (max-heap via negative freq + tie-break stored in HeapItem)

    Assumes you have HeapItem defined like:
        class HeapItem:
            def __init__(self, neg_freq, pair_bytes, pair): ...
            def __lt__(self, other): ...
    """
    a, b = target_pair

    # Start from full counters so unaffected words remain.
    new_word_counter: Counter = Counter(word_counter)
    updated_pair_counter: Counter = pair_counter.copy()
    changed_pairs: set[tuple[int, int]] = set()

    affected_words = list(pair_to_words.get(target_pair, set()))

    for w in affected_words:
        freq = word_counter.get(w, 0)
        if freq <= 0:
            continue

        L = len(w)
        if L < 2:
            continue

        # (A) Remove the old word from the corpus counts.
        new_word_counter[w] -= freq
        if new_word_counter[w] <= 0:
            del new_word_counter[w]

        # (B) Subtract ALL old adjacent pairs for this word + remove old word from index.
        prev = w[0]
        for cur in w[1:]:
            p = (prev, cur)
            updated_pair_counter[p] -= freq
            changed_pairs.add(p)

            s = pair_to_words.get(p)
            if s is not None:
                s.discard(w)
                if not s:
                    del pair_to_words[p]

            prev = cur

        # (C) Build merged word (greedy left-to-right, same as standard BPE).
        out: list[int] = []
        i = 0
        while i < L:
            if i + 1 < L and w[i] == a and w[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(w[i])
                i += 1

        new_w = tuple(out)

        # (D) Add merged word to corpus counts.
        new_word_counter[new_w] += freq

        # (E) Add ALL new adjacent pairs for merged word + add merged word into index.
        if len(out) >= 2:
            prev = out[0]
            for cur in out[1:]:
                p = (prev, cur)
                updated_pair_counter[p] += freq
                changed_pairs.add(p)
                pair_to_words.setdefault(p, set()).add(new_w)
                prev = cur

    # (F) Clean up non-positive pair counts.
    for p in list(updated_pair_counter.keys()):
        if updated_pair_counter[p] <= 0:
            del updated_pair_counter[p]

    # (G) Push updated frequencies for changed pairs into heap (skip non-positive).
    if pair_heap is not None:
        for p in changed_pairs:
            f = updated_pair_counter.get(p, 0)
            if f > 0:
                heapq.heappush(pair_heap, HeapItem(-f, (vocab[p[0]], vocab[p[1]]), p))

    return dict(new_word_counter), updated_pair_counter, pair_heap, pair_to_words
