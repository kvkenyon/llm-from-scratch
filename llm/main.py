import json
import time
from pathlib import Path

from llm.common import gpt2_bytes_to_unicode
from llm.tokenizer import tokenize


def run_train_bpe_owt():
    filepath = Path(__file__).parent.parent.resolve() / "data" / "owt_train.txt"

    start_time = time.perf_counter()

    vocab, merges = tokenize(str(filepath), vocab_size=32_000, special_tokens=["<|endoftext|>"])

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")

    vocab_serial = make_vocab_serializable(vocab)
    serialized = json.dumps(vocab_serial)
    with open("vocab_owt_train.json", "x") as f:
        f.write(serialized)

    merges_nice = make_merges_readable(merges)
    with open("owt_train_merges.txt", "x") as f:
        f.write(merges_nice)


def main():
    run_train_bpe_owt()
    # filepath = Path(__file__).parent.parent.resolve() / "data" / "TinyStoriesV2-GPT4-train.txt"
    # # filepath = Path(__file__).parent.parent.resolve() / "data" / "corpus.en"
    #
    # start_time = time.perf_counter()
    # vocab, merges = tokenize(str(filepath), vocab_size=10000, special_tokens=["<|endoftext|>"])
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Execution time: {elapsed_time:.6f} seconds")
    #
    # assert len(vocab) == 10_000, "Vocab not sized right"
    #
    # vocab_serial = make_vocab_serializable(vocab)
    #
    # serialized = json.dumps(vocab_serial)
    #
    # with open("vocab.json", "x") as f:
    #     f.write(serialized)
    #
    # merges_nice = make_merges_readable(merges)
    # with open("merges.txt", "x") as f:
    #     f.write(merges_nice)
    #
    #


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


if __name__ == "__main__":
    main()
