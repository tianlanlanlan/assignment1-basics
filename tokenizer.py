from cs336_basics.bpe import Tokenizer, BPE
from pathlib import Path


def main():
    train_data_txt = "data/TinyStoriesV2-GPT4-train.txt"
    valid_data_txt = "data/TinyStoriesV2-GPT4-valid.txt"

    assert train_data_txt.endswith(".txt")
    assert valid_data_txt.endswith(".txt")

    vocab_json = train_data_txt.removesuffix(".txt") + ".vocab.json"
    merges_txt = train_data_txt.removesuffix(".txt") + ".merges.txt"

    if not Path(vocab_json).exists() or not Path(merges_txt).exists():
        bpe = BPE()
        bpe.train(train_data_txt, ["<|endoftext|>"], 10_000)
        bpe.save_vocab_json(vocab_json)
        bpe.save_merges_txt(merges_txt)


if __name__ == "__main__":
    main()
