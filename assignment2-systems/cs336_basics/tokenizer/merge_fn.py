import heapq
from collections import Counter, defaultdict


def get_most_frequent_pair(
    pair_counter: dict[tuple[int, int], int], vocab: dict[int, bytes]
) -> tuple[int, int]:
    max_freq = max(pair_counter.values())

    candidates = [
        (pair, (vocab[pair[0]], vocab[pair[1]])) for pair, freq in pair_counter.items() if freq == max_freq
    ]
    candidates.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)

    return candidates[0][0]


def get_new_word(
    word: tuple[int, ...],
    target_pair: tuple[int, int],
    new_id: int,
) -> tuple[int, ...]:
    a, b = target_pair
    new_word = []
    i = 0

    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)


def need_merge(word: tuple[int, ...], target_pair: tuple[int, int]) -> bool:
    if len(word) < 2:
        return False

    a, b = target_pair
    i = 0

    while i + 1 < len(word):
        if word[i] == a and word[i + 1] == b:
            return True
        i += 1
    return False


# Version 1: Simple pair merging without efficient updates
def merge_pairs(
    word_counter: dict[tuple[int, ...], int],
    target_pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int, ...], int], dict[tuple[int, int], int]]:
    new_word_counter: defaultdict[tuple[int, ...], int] = defaultdict(int)
    updated_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    for word, freq in word_counter.items():
        new_word = get_new_word(word, target_pair, new_id)
        new_word_counter[tuple(new_word)] += freq

        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                updated_pair_counts[pair] += freq

    return new_word_counter, updated_pair_counts


# Version 2: Incremental pair merging with efficient updates
def merge_pairs_incremental(
    word_counter: dict[tuple[int, ...], int],
    pair_counter: Counter,
    target_pair: tuple[int, int],
    new_id: int,
) -> tuple[dict[tuple[int, ...], int], Counter]:
    new_word_counter: defaultdict[tuple[int, ...], int] = defaultdict(int)

    for word, freq in word_counter.items():
        if not need_merge(word, target_pair):
            new_word_counter[word] += freq
            continue

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counter[pair] -= freq
            if pair_counter[pair] <= 0:
                del pair_counter[pair]

        # Update word
        new_word = get_new_word(word, target_pair, new_id)
        new_word_counter[new_word] += freq
        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_counter[pair] += freq

    return new_word_counter, pair_counter


# Version 3: Using heap for pair selection
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
    return heap


def pop_most_frequent_pair(heap, pairs_counter: Counter) -> tuple[int, int]:
    while heap:
        item = heap[0]  # Peek at the top item
        neg_f = item.neg_freq
        pair = item.pair
        cur_f = pairs_counter.get(pair, 0)
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
        if not need_merge(w, target_pair):
            new_word_counter[w] += freq
            continue

        # (1) subtract old adjacent pairs for this word
        for i in range(L - 1):
            pair = (w[i], w[i + 1])
            updated_pair_counter[pair] -= freq
            if updated_pair_counter[pair] <= 0:
                del updated_pair_counter[pair]
            changed_pairs.add(pair)

        # (2) build merged word
        new_word = get_new_word(w, target_pair, new_id)
        new_word_counter[new_word] += freq

        # (3) add new adjacent pairs for merged word
        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                updated_pair_counter[pair] += freq
                changed_pairs.add(pair)

    # Update the heap with new pair frequencies
    if pair_heap is not None:
        for pair in changed_pairs:
            freq = updated_pair_counter.get(pair, 0)
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
    # Start from full counters so unaffected words remain.
    new_word_counter: Counter = Counter(word_counter)
    updated_pair_counter: Counter = pair_counter.copy()
    changed_pairs: set[tuple[int, int]] = set()

    # Get all words that contain the target pair.
    affected_words = list(pair_to_words.get(target_pair, set()))

    for w in affected_words:
        freq = word_counter.get(w, 0)
        if freq <= 0 or len(w) < 2:
            continue

        # 1. Remove the old word from the corpus counts.
        new_word_counter[w] -= freq
        if new_word_counter[w] <= 0:
            del new_word_counter[w]

        # 2. Subtract ALL old adjacent pairs for this word + remove old word from index.
        for i in range(len(w) - 1):
            pair = (w[i], w[i + 1])
            updated_pair_counter[pair] -= freq
            changed_pairs.add(pair)

            s = pair_to_words.get(pair)
            if s is not None:
                s.discard(w)
                if not s:
                    del pair_to_words[pair]

        # 3. Build merged word (greedy left-to-right, same as standard BPE).
        new_word = get_new_word(w, target_pair, new_id)
        new_word_counter[new_word] += freq

        # 4. Add ALL new adjacent pairs for merged word + add merged word into index.
        if len(new_word) >= 2:
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                updated_pair_counter[pair] += freq
                changed_pairs.add(pair)
                pair_to_words.setdefault(pair, set()).add(new_word)

    # 5. Push updated frequencies for changed pairs into heap (skip non-positive).
    if pair_heap is not None:
        for p in changed_pairs:
            f = updated_pair_counter.get(p, 0)
            if f > 0:
                heapq.heappush(pair_heap, HeapItem(-f, (vocab[p[0]], vocab[p[1]]), p))

    return dict(new_word_counter), updated_pair_counter, pair_heap, pair_to_words
