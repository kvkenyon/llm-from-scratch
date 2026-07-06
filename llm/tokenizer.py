import os
from collections import Counter, defaultdict
from functools import partial
from itertools import batched, pairwise
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

DEBUG = False


def init_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([b]) for i, b in enumerate(bytes(range(256)))}
    for i, special_token in enumerate(special_tokens, 1):
        vocab[i + 255] = special_token.encode("utf-8")
    return vocab


def assert_no_special_characters(pretokens):
    invalid_bytes = [b.encode("utf-8") for b in "<|>"]
    for pretoken, _ in pretokens.items():
        for pt in pretoken:
            if pt in invalid_bytes:
                assert False, f"illegal char for {pt}"


def tokenize(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretoken_counts = pretokenize(input_path, special_tokens)
    vocab = init_vocabulary(special_tokens)

    merges = []

    pretokens = [(pretoken, count) for pretoken, count in pretoken_counts.items()]

    byte_pair_cache = defaultdict(set)

    byte_pair_freq = Counter()
    for i, (pretoken, count) in enumerate(pretokens):
        for x, y in pairwise(pretoken):
            byte_pair = (x, y)
            byte_pair_freq[byte_pair] += count
            byte_pair_cache[byte_pair].add(i)

    while len(vocab) < vocab_size:
        merge = max(byte_pair_freq, key=lambda k: (byte_pair_freq[k], k))

        merges.append(merge)

        for pid in list(byte_pair_cache[merge]):
            pretoken, count = pretokens[pid]
            i = 1
            if DEBUG:
                locs = find_byte_pair(pretoken, merge)
                if not locs:
                    raise ValueError(f"byte pair {merge} not found in {pretoken}")

            new_pretoken = update_pretoken(pid, pretoken, count, merge, byte_pair_freq, byte_pair_cache)
            pretokens[pid] = (tuple(new_pretoken), count)

        del byte_pair_cache[merge]
        del byte_pair_freq[merge]
        vocab[len(vocab)] = merge[0] + merge[1]

    return vocab, merges


def find_byte_pair(pretoken: list[bytes, ...], merge: tuple[bytes, bytes]) -> list[int]:
    i = 1
    locs = []
    while i < len(pretoken):
        x, y = pretoken[i - 1], pretoken[i]
        if (x, y) == merge:
            locs.append(i - 1)
        i += 1
    return locs


def update_pretoken(
    pid: int,
    pretoken: tuple[bytes, ...],
    pretoken_freq: int,
    merge: tuple[bytes, bytes],
    byte_pair_freq,
    byte_pair_cache,
):
    # pretoken [B, A, A, B, A], merge = [A,A]

    # [B, AA, B, A]
    new_pretoken = merge_tokens(pretoken, merge)

    # [B_A, A_A, A_B, B_A]
    old_pairs = Counter([(a, b) for a, b in pairwise(pretoken)])
    # [B_AA, AA_B, B_A]
    new_pairs = Counter([(a, b) for a, b in pairwise(new_pretoken)])

    # old_pairs = {B_A:2, A_A:1, A_B:1}
    # new_pairs = {B_A:1, A_A:0 ,A_B:0, B_AA: 1, AA_B: 1}

    # {B_A: 1, A_A: 1, A_B: 1, B_AA: -1, AA_B: -1}
    old_pairs.subtract(new_pairs)

    for byte_pair, delta in old_pairs.items():
        if byte_pair == merge:
            continue
        byte_pair_freq[byte_pair] += pretoken_freq * delta * -1

        if byte_pair not in new_pairs:
            byte_pair_cache[byte_pair].remove(pid)
        else:
            byte_pair_cache[byte_pair].add(pid)

    return new_pretoken


def merge_tokens(pretoken, merge):
    new_pretoken = []
    i = 1
    while i < len(pretoken):
        byte_pair = (pretoken[i - 1], pretoken[i])
        if byte_pair == merge:
            new_pretoken.append(byte_pair[0] + byte_pair[1])
            i += 2
        else:
            new_pretoken.append(byte_pair[0])
            i += 1
        if i == len(pretoken):
            new_pretoken.append(pretoken[-1])
    return new_pretoken


def pretokenize(input_path: str, special_tokens: list[str], desired_num_chunks=4):
    assert special_tokens, f"Require one special token to split the corpus into {desired_num_chunks}"
    assert desired_num_chunks > 0, "desired_num_chunks must be positive"
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_tokens[0].encode("utf-8"))
        bounds = zip(boundaries[:-1], boundaries[1:])
        chunks = [_get_chunk(f, bound) for bound in bounds]
        f = partial(_pretokenize, special_tokens=special_tokens)
        with Pool(num_processes) as pool:
            counters = pool.map(f, chunks)
        pretokens = Counter()
        for counter in counters:
            pretokens += counter
    return pretokens


def _pretokenize(chunk: str, pat: str = PAT, special_tokens: list[str] | None = None):
    counts = Counter()

    def _pretokenize_subchunk(subchunk: str):
        for batch in batched(re.finditer(pat, subchunk), 10):
            counts.update([tuple([bytes([b]) for b in m.group().encode("utf-8")]) for m in batch])

    if special_tokens:
        subchunks = _split_chunk(chunk, special_tokens)
        for subchunk in subchunks:
            _pretokenize_subchunk(subchunk)
    else:
        _pretokenize_subchunk(chunk)

    return counts


def _get_chunk(f, bound):
    start, end = bound
    f.seek(start)
    return f.read(end - start).decode("utf-8", errors="ignore")


def _split_chunk(chunk: str, special_tokens: list[str]) -> str:
    escaped_special_tokens = [re.escape(special_token) for special_token in special_tokens]
    pattern = "|".join(escaped_special_tokens)
    return re.split(pattern, chunk)


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
