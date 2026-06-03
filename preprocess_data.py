import numpy as np
from multiprocessing import Pool
from pathlib import Path
from cs336_basics.bpe import Tokenizer

DATA_DIR = "data"
TRAIN_TXT = f"{DATA_DIR}/TinyStoriesV2-GPT4-train.txt"
VALID_TXT = f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.txt"


def _tokenize_chunk(args):
    """Tokenize lines [start_line, end_line) of a text file."""
    input_path, start_line, end_line, chunk_idx, vocab_json, merges_txt = args
    tokenizer = Tokenizer.from_files(vocab_json, merges_txt, special_tokens=["<|endoftext|>"])
    eot_id = tokenizer.encode("<|endoftext|>")[0]

    all_ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if i >= end_line:
                break
            line = line.strip()
            if line:
                all_ids.extend(tokenizer.encode(line))
            all_ids.append(eot_id)
    return np.array(all_ids, dtype=np.uint16)


def tokenize_file_parallel(input_path, output_path, num_processes=10):
    print(f"Tokenizing {input_path} with {num_processes} processes...")

    # Count lines
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Split into chunks at line boundaries
    chunk_size = total_lines // num_processes
    chunk_args = []
    for i in range(num_processes):
        start = i * chunk_size
        end = total_lines if i == num_processes - 1 else (i + 1) * chunk_size
        chunk_args.append(
            (
                input_path,
                start,
                end,
                i,
                f"{DATA_DIR}/TinyStoriesV2-GPT4-train.vocab.json",
                f"{DATA_DIR}/TinyStoriesV2-GPT4-train.merges.txt",
            )
        )

    print(f"Processing {total_lines} lines in {num_processes} chunks...")
    with Pool(num_processes) as pool:
        results = pool.map(_tokenize_chunk, chunk_args)

    # Count total tokens and write via memmap
    total_tokens = sum(len(arr) for arr in results)
    print(f"Total tokens: {total_tokens:,}")

    arr = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    offset = 0
    for chunk_arr in results:
        arr[offset : offset + len(chunk_arr)] = chunk_arr
        offset += len(chunk_arr)
    arr.flush()
    del arr

    file_size = Path(output_path).stat().st_size
    print(f"Saved to {output_path} ({file_size / 1e9:.2f} GB)")


def main():
    tokenize_file_parallel(TRAIN_TXT, Path(f"{DATA_DIR}/TinyStoriesV2-GPT4-train.bin"))
    tokenize_file_parallel(VALID_TXT, Path(f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.bin"))


if __name__ == "__main__":
    main()
