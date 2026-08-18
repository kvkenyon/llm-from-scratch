from __future__ import annotations

import itertools
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import partial
from itertools import batched, pairwise
from multiprocessing import Pool
from pathlib import Path
from typing import BinaryIO

import regex as re
from tqdm import tqdm

from .common import load_merges, load_vocab, stream_dataset

NUM_PROCESSES = 4
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

DEBUG = False


def init_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([b]) for i, b in enumerate(bytes(range(256)))}
    for i, special_token in enumerate(special_tokens, 1):
        vocab[i + 255] = special_token.encode("utf-8")
    return vocab


def assert_no_special_characters(pretokens):
    invalid_bytes = [b.encode("utf-8") for b in "<|>"]
    for pretoken in pretokens:
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

    with tqdm(total=vocab_size - len(vocab), desc="tokenize") as pbar:
        while len(vocab) < vocab_size:
            merge = max(byte_pair_freq, key=lambda k: (byte_pair_freq[k], k))

            merges.append(merge)

            for pid in list(byte_pair_cache[merge]):
                pretoken, count = pretokens[pid]

                new_pretoken = update_pretoken(pid, pretoken, count, merge, byte_pair_freq, byte_pair_cache)
                pretokens[pid] = (tuple(new_pretoken), count)

            del byte_pair_cache[merge]
            del byte_pair_freq[merge]
            vocab[len(vocab)] = merge[0] + merge[1]
            pbar.update(1)

    return vocab, merges


def find_byte_pair(pretoken: list[bytes], merge: tuple[bytes, bytes]) -> list[int]:
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


def pretokenize(filepath: str, special_tokens: list[str], desired_num_chunks=12):
    assert special_tokens, f"Require one special token to split the corpus into {desired_num_chunks}"
    assert desired_num_chunks > 0, "desired_num_chunks must be positive"
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_tokens[0].encode("utf-8"))
        bounds = itertools.pairwise(boundaries)
        fn = partial(_pretokenize, filepath=filepath, special_tokens=special_tokens)
        with Pool(NUM_PROCESSES) as pool:
            counters = pool.map(fn, list(bounds))
        pretokens = Counter()
        for counter in counters:
            pretokens += counter
    return pretokens


def _pretokenize(bound: tuple[int, int], filepath: str, pat: str = PAT, special_tokens: list[str] | None = None):
    counts = Counter()

    def _pretokenize_subchunk(subchunk: str):
        for batch in batched(re.finditer(pat, subchunk), 10):
            counts.update([tuple([bytes([b]) for b in m.group().encode("utf-8")]) for m in batch])

    with open(filepath, "rb") as f:
        chunk = _get_chunk(f, bound)
        if special_tokens:
            subchunks = _split_chunk(chunk, special_tokens)
            for subchunk in subchunks:
                _pretokenize_subchunk(subchunk)
        else:
            _pretokenize_subchunk(chunk)

    return counts


def _get_chunk(f: BinaryIO, bound: tuple[int, int]):
    start, end = bound
    f.seek(start)
    return f.read(end - start).decode("utf-8", errors="ignore")


def _split_chunk(chunk: str, special_tokens: list[str]):
    escaped_special_tokens = [re.escape(special_token) for special_token in special_tokens]
    pattern = "|".join(escaped_special_tokens)
    return re.splititer(pattern, chunk)


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


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.inv_vocab = {v: i for i, v in self.vocab.items()}

        # append special tokens to vocab
        max_id = max(self.vocab, key=self.vocab.get)  # type: ignore
        next_id = max_id + 1
        for st in special_tokens or []:
            st_utf8 = st.encode("utf-8")
            if st_utf8 not in self.inv_vocab:
                self.vocab[next_id] = st_utf8
                self.inv_vocab[st_utf8] = next_id
                next_id += 1

        self.merges = merges

        self.merges_index = {m: i for i, m in enumerate(merges)}

        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> Tokenizer:
        vocab = load_vocab(vocab_filepath, True)
        merges = load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _find_next_merge(self, pretoken) -> tuple[bytes, bytes] | None:
        next_merge_idx = sys.maxsize
        for i in range(1, len(pretoken)):
            byte_pair = (pretoken[i - 1], pretoken[i])
            if byte_pair in self.merges_index:
                candidate_idx = self.merges_index[byte_pair]
                next_merge_idx = min(candidate_idx, next_merge_idx)
        return self.merges[next_merge_idx] if next_merge_idx != sys.maxsize else None

    def encode(self, text: str):
        pretokens = self._pretokenize(text)

        results = []
        for pretoken in pretokens:
            if pretoken in self.special_tokens:
                results.append(self.inv_vocab[pretoken.encode("utf-8")])
                continue
            while (merged := self._find_next_merge(pretoken)) is not None:
                pretoken = merge_tokens(pretoken, merged)

            for token in pretoken:
                results.append(self.inv_vocab[token])

        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        result = b"".join([self.vocab[id] for id in ids])
        return result.decode("utf-8", "replace")

    def _pretokenize(self, text: str) -> list[tuple[bytes, ...]]:
        results = []
        if self.special_tokens:
            chunks = self._split_text(text)
            for chunk in chunks:
                if chunk in self.special_tokens:
                    results.append(chunk)
                    continue
                for batch in batched(re.finditer(PAT, chunk), 10):
                    pretokens = [tuple([bytes([b]) for b in m.group().encode("utf-8")]) for m in batch]
                    results.extend(pretokens)
            return results

        for batch in batched(re.finditer(PAT, text), 10):
            pretokens = [tuple([bytes([b]) for b in m.group().encode("utf-8")]) for m in batch]
            results.extend(pretokens)
        return results

    def _split_text(self, text: str):
        escaped = [re.escape(st) for st in self.special_tokens]
        sorted_escaped = sorted(escaped, key=len, reverse=True)
        pattern = "|".join(sorted_escaped)
        pattern = f"({pattern})"
        return [chunk for chunk in re.split(pattern, text) if chunk]


def tokenizer_compression_ratio(tokenizer: Tokenizer, dataset: str):
    # bytes / token

    filepath = Path(__file__).parent.parent.resolve() / "data" / dataset

    samples = []

    for i, sample in enumerate(stream_dataset(filepath)):
        if i == 10:
            break
        samples.append(sample)

    encoded = []
    for sample in samples:
        encoded_sample = tokenizer.encode(sample)
        encoded.append(encoded_sample)

    comp_ratios = []
    for tokens, original in zip(encoded, samples):
        comp_ratio = len(original.encode("utf-8")) / len(tokens)
        comp_ratios.append(comp_ratio)

    return comp_ratios


def load_benchmark_sample(dataset: str, target_bytes: int = 10 * 1024**2):
    filepath = Path("data") / dataset
    samples = []
    total_bytes = 0

    for text in stream_dataset(filepath):
        samples.append(text)
        total_bytes += len(text.encode("utf-8"))

        if total_bytes >= target_bytes:
            break

    return samples, total_bytes


def benchmark_tokenizer(
    tokenizer,
    dataset: str,
    target_bytes: int = 10 * 1024**2,
    min_run_seconds: float = 1.0,
    runs: int = 5,
    show_progress: bool = True,
):
    samples, sample_bytes = load_benchmark_sample(dataset, target_bytes)

    if not samples:
        raise ValueError(f"No samples found in data/{dataset}")

    # A small warm-up avoids encoding the entire (potentially slow) sample twice.
    tokenizer.encode(samples[0])
    sample_sizes = [(text, len(text.encode("utf-8"))) for text in samples]

    results = []
    total_token_count = 0

    with tqdm(
        total=runs * sample_bytes,
        desc="Benchmarking encode",
        unit="B",
        unit_scale=True,
        disable=not show_progress,
    ) as pbar:
        for run in range(runs):
            processed_bytes = 0
            token_count = 0
            elapsed = 0.0

            while True:
                for text, text_bytes in sample_sizes:
                    start = time.perf_counter()
                    token_count += len(tokenizer.encode(text))
                    elapsed += time.perf_counter() - start
                    processed_bytes += text_bytes
                    pbar.update(text_bytes)

                if elapsed >= min_run_seconds:
                    break

                # This run needs another pass to reach the minimum duration.
                pbar.total += sample_bytes
                pbar.refresh()

            rate = processed_bytes / elapsed
            results.append(rate)
            total_token_count += token_count
            pbar.set_postfix(run=run + 1, MiB_s=f"{rate / 1024**2:.2f}")

    median_rate = statistics.median(results)

    return {
        "sample_mib": sample_bytes / 1024**2,
        "bytes_per_second": median_rate,
        "mib_per_second": median_rate / 1024**2,
        "runs_mib_per_second": [rate / 1024**2 for rate in results],
        "tokens_processed": total_token_count,
    }
