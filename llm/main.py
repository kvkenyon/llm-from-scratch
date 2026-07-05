from pathlib import Path

from llm.tokenizer import tokenize


def main():
    filepath = Path(__file__).parent.parent.resolve() / "data" / "corpus.en"
    vocab, _ = tokenize(str(filepath), vocab_size=400, special_tokens=["<|endoftext|>"])

    assert len(vocab) == 400, "Vocab not sized right"


if __name__ == "__main__":
    main()
