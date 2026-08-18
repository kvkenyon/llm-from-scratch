import json
import time
from pathlib import Path

from llm.common import gpt2_bytes_to_unicode
from llm.tokenizer import tokenize


def main(dataset: str, vocab_size: int, special_tokens: list[str] | None = None):
    special_tokens = ["<|endoftext|>"] if not special_tokens else special_tokens
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
