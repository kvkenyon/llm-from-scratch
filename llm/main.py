import json
import time
from pathlib import Path

from llm.common import gpt2_bytes_to_unicode
from llm.tokenizer import tokenize


def main():
    filepath = Path(__file__).parent.parent.resolve() / "data" / "TinyStoriesV2-GPT4-train.txt"
    # filepath = Path(__file__).parent.parent.resolve() / "data" / "corpus.en"

    start_time = time.perf_counter()
    vocab, merges = tokenize(str(filepath), vocab_size=10000, special_tokens=["<|endoftext|>"])
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")

    assert len(vocab) == 10_000, "Vocab not sized right"

    vocab_serial = make_vocab_serializable(vocab)
    import pprint

    pprint.pprint(vocab_serial)

    serialized = json.dumps(vocab_serial)

    with open("vocab.json", "x") as f:
        f.write(serialized)


def bytes_to_readable(bs: bytes) -> str:
    gpt2_bytes_decoder = gpt2_bytes_to_unicode()
    return "".join([gpt2_bytes_decoder[b] for b in list(bs)])


def make_vocab_serializable(vocab: dict[int, bytes]) -> None:
    d = {}
    for i, vocab_item in vocab.items():
        s = bytes_to_readable(vocab_item)
        d[s] = i
    return d


if __name__ == "__main__":
    main()
