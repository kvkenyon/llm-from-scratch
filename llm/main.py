import json
import time
from pathlib import Path

from llm.common import gpt2_bytes_to_unicode
from llm.tokenizer import Tokenizer, tokenize


def stream_dataset(filepath: str, split_on: str = "<|endoftext|>", chunk_size: int = 65_536):
    with open(filepath, encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            buffer += chunk

            parts = buffer.split(split_on)

            for part in parts[:-1]:
                doc = part.strip()
                if doc:
                    yield doc

            buffer = parts[-1]

        final_doc = buffer.strip()
        if final_doc:
            yield final_doc


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


def main(dataset: str, vocab_size: int, special_tokens: list[str] | None = ["<|endoftext|>"]):

    filepath = Path(__file__).parent.parent.resolve() / "data" / dataset
    start_time = time.perf_counter()

    vocab, merges = tokenize(str(filepath), vocab_size=vocab_size, special_tokens=special_tokens)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")

    dataset_name = dataset.split(".")[0]

    vocab_serial = make_vocab_serializable(vocab)
    serialized = json.dumps(vocab_serial)

    with open(f"vocab_{dataset_name}.json", "x") as f:
        f.write(serialized)

    merges_nice = make_merges_readable(merges)
    with open(f"merges_{dataset_name}.txt", "x") as f:
        f.write("".join(merges_nice))


def bytes_to_readable(bs: bytes) -> str:
    gpt2_bytes_decoder = gpt2_bytes_to_unicode()
    return "".join([gpt2_bytes_decoder[b] for b in list(bs)])


def make_merges_readable(merges: list[tuple[bytes, bytes]]) -> list[str]:
    results = []
    for a, b in merges:
        a_str = bytes_to_readable(a)
        b_str = bytes_to_readable(b)
        r = f"{a_str} {b_str}\n"
        results.append(r)
    return results


def make_vocab_serializable(vocab: dict[int, bytes]) -> dict[str, int]:
    d = {}
    for i, vocab_item in vocab.items():
        s = bytes_to_readable(vocab_item)
        d[s] = i
    return d
