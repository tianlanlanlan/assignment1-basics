from collections import Counter
import os
import regex as re
import queue
from collections import defaultdict

from multiprocessing import Pool


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


BytesPair = tuple[bytes, bytes]


class BytesPairFreq:
    def __init__(self, token_freq: int = 0, locations: set = None):
        self.token_freq = token_freq
        self.locations = locations if locations is not None else set()


class BytePairHeapItem:
    def __init__(self, token_pair: BytesPair, token_freq: BytesPairFreq):
        self.token_pair = token_pair
        self.token_freq = token_freq

    def __repr__(self):
        return f"({self.token_freq}, {self.token_pair})"

    # 构建大顶堆， return True 靠近堆顶，return False 靠近堆底
    def __lt__(self, other: "BytePairHeapItem"):
        if self.token_freq.token_freq == other.token_freq.token_freq:
            return self.token_pair > other.token_pair
        else:
            return self.token_freq.token_freq > other.token_freq.token_freq

    @staticmethod
    def ConstructPriorityQueue(token_pair_freq: dict[BytesPair, BytesPairFreq]) -> queue.PriorityQueue:
        max_heap = queue.PriorityQueue()
        for token_bytes_tuple, token_freq in token_pair_freq.items():
            max_heap.put(BytePairHeapItem(token_pair=token_bytes_tuple, token_freq=token_freq))
        return max_heap


def TestBytePairHeapItem():
    def construct_priority_queue(token_pair_freq: Counter[tuple[bytes, bytes]]) -> queue.PriorityQueue:
        max_heap = queue.PriorityQueue()
        for tok_bytes_tuple, tok_freq in token_pair_freq.items():
            max_heap.put((-tok_freq, tok_bytes_tuple))
        return max_heap

    pair_bytes_freq: Counter[tuple[bytes, bytes]] = {(b"a", b"b"): 3, (b"a", b"c"): 3, (b"c", b"d"): 100}
    print(f"pair_bytes_freq: {pair_bytes_freq}")

    pqueue = construct_priority_queue(pair_bytes_freq)

    pqueue.get()
    top1 = pqueue.get()
    assert top1 == (-3, (b"a", b"b")), f"Pop: {top1}"
    pqueue.get()
    # print(f"pqueue top: {pqueue.get()} {pqueue.get()} {pqueue.get()}")

    pqueue.put(BytePairHeapItem((b"a", b"b"), 3))
    pqueue.put(BytePairHeapItem((b"a", b"c"), 3))
    pqueue.put(BytePairHeapItem((b"b", b"c"), 3))
    pqueue.put(BytePairHeapItem((b"\x6f", b"\x8f"), 3))

    top1: BytePairHeapItem = pqueue.get()
    assert top1.token_pair == (b"\x6f", b"\x8f")


def process_file_chunk(
    input_file_path: str, start_boundary: int, end_boundary: int, pattern_include_special_token: str
) -> Counter[str]:
    # Step2. 按照 chunk_boundaries_after 并行处理, 过滤 special tokens + 按照 PAT 分词
    token_freq = Counter()
    print(f"Processing [{start_boundary}, {end_boundary}] ...")

    with open(input_file_path, "rb") as f:
        f.seek(start_boundary)
        chunk_bytes: bytes = f.read(end_boundary - start_boundary)
        chunk_string: str = chunk_bytes.decode("utf-8")  # Bytes to string

        iters = re.finditer(pattern_include_special_token, chunk_string)
        for it in iters:
            token_freq[it.group()] += 1

        return token_freq


def pretokenization(input_filepath: str, special_tokens: list[str], desired_num_chunks: int = 8) -> Counter:
    token_freq = Counter()

    # Convert special tokens from string into bytes
    special_tokens_bytes: list[bytes] = [tokstr.encode("utf-8") for tokstr in special_tokens]
    max_special_tokens_len: int = max(len(tok) for tok in special_tokens_bytes)

    chunk_boundaries_before: list[int] = []
    chunk_boundaries_after: list[int] = []

    with open(input_filepath, "rb") as f:
        f.seek(0, os.SEEK_END)
        filebytesize = f.tell()
        f.seek(0, os.SEEK_SET)
        print(f"filebytesize: {filebytesize / (1024.0 * 1024.0):.2f} MB")

        chunksize = filebytesize // desired_num_chunks

        chunk_boundaries_before = [i * chunksize for i in range(desired_num_chunks + 1)]
        chunk_boundaries_before[-1] = filebytesize

        # Adjust middle index
        mini_chunksize: int = 4096
        for i in range(1, len(chunk_boundaries_before)):
            offset = chunk_boundaries_before[i]
            f.seek(offset, os.SEEK_SET)
            last_chunk_remainder: bytes = b""

            while True:
                filechunk_raw: bytes = f.read(mini_chunksize)
                # Found file EOF
                if filechunk_raw == b"":
                    chunk_boundaries_before[i] = filebytesize
                    break
                filechunksize = len(filechunk_raw)

                # 为防止截断 speical token，拼接上个 chunk 的后 (max_special_tokens_len - 1) 个字节再开始搜索
                filechunk = last_chunk_remainder + filechunk_raw

                # Interate all speical token
                min_found_at: int = len(filechunk)
                for tok in special_tokens_bytes:
                    found_at: int = filechunk.find(tok)
                    if found_at != -1:
                        min_found_at = min(min_found_at, found_at)
                if min_found_at < filechunksize:
                    chunk_boundaries_before[i] = offset + min_found_at - len(last_chunk_remainder)
                    break

                # Continue match
                offset += filechunksize

                # Update last_chunk_remainder
                last_chunk_remainder = filechunk[-(max_special_tokens_len - 1) :]

        chunk_boundaries_after = sorted(set(chunk_boundaries_before))
        print(f"chunk_boundaries_after: {chunk_boundaries_after}")

    # Step1. 并行正则匹配，不剔除 special token，一起统计
    special_tokens_join: str = "|".join(re.escape(tok) for tok in special_tokens)
    pattern_include_special_token: str = f"{special_tokens_join}|{PAT}"

    start_boundaries = [chunk_boundaries_after[i] for i in range(len(chunk_boundaries_after) - 1)]
    end_boundaries = [chunk_boundaries_after[i] for i in range(1, len(chunk_boundaries_after))]

    args = [
        (input_filepath, start_boundary, end_boundary, pattern_include_special_token)
        for start_boundary, end_boundary in zip(start_boundaries, end_boundaries)
    ]
    with Pool(desired_num_chunks) as pool:
        results: list[Counter] = pool.starmap(process_file_chunk, args)

    for result in results:
        token_freq.update(result)

    return token_freq


def merge(
    token_freq_str: Counter[str], vocab_size: int, spepcial_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256, f"Require {vocab_size} >= 256"
    vocab: dict[int, bytes] = defaultdict()
    merges: list[tuple[bytes, bytes]] = []

    # Counter[str] -> Counter[bytes]
    token_freq: Counter[tuple[bytes, ...]] = Counter()
    token_pair_freq: dict[BytesPair, BytesPairFreq] = defaultdict(BytesPairFreq)
    for tok_str, tok_freq in token_freq_str.items():
        tok_bytes: bytes = tok_str.encode("utf-8")
        tok_bytes_tuple = tuple(
            [bytes([b]) for b in tok_bytes]
        )  # 坑点: python 里 bytes 对象 tok_bytes[i] 返回的类型，不是 bytes，而是 int
        token_freq[tok_bytes_tuple] = tok_freq

        for i in range(1, len(tok_bytes)):
            bytes_pair = (
                bytes([tok_bytes[i - 1]]),
                bytes([tok_bytes[i]]),
            )  # 坑点: python 里 bytes 对象 tok_bytes[i] 返回的类型，不是 bytes，而是 int，所以需要转成 bytes。转的时候需要 bytes([int]) 而不是 bytes(int)
            token_pair_freq[bytes_pair].token_freq += tok_freq
            token_pair_freq[bytes_pair].locations.add(tok_bytes_tuple)  # 存引用，不拷贝

    # print(f"token_freq: {token_freq}")
    # print(f"token_pair_freq: {token_pair_freq}")
    max_heap: queue.PriorityQueue = BytePairHeapItem.ConstructPriorityQueue(token_pair_freq)
    # print(f"max_heap: {max_heap}")

    # 构造 vocabulary, 加入 256 (0 ~ 255) 个字节 + special_tokens
    for num in range(256):
        num_byte = bytes([num])  # Convert int number into bytes
        vocab[num_byte] = num

    vocab_token_id: int = 256
    for tok_bytes in [tok.encode("utf-8") for tok in spepcial_tokens]:
        vocab[tok_bytes] = vocab_token_id
        vocab_token_id += 1

    while vocab_token_id < vocab_size:
        max_freq_pair: BytePairHeapItem = max_heap.get()
        print(f"max_freq_pair: {max_freq_pair.token_pair}")

        # Add into vocab and merge
        max_freq_bytes: BytesPair = max_freq_pair.token_pair
        # assert len(max_freq_bytes) == 2
        vocab[b"".join(max_freq_bytes)] = vocab_token_id
        vocab_token_id += 1
        merges.append(max_freq_pair)

        # Merge
        token_set_need_merge: set = max_freq_pair.token_freq.locations
        print(f"tokens_need_merge: {token_set_need_merge}")

        for token_need_merge in token_set_need_merge.copy():  # 循环内会从 set 中删除, 所以迭代副本而不是原始 set
            merged_tokens_list: list[bytes] = []
            tokens_need_merge_len: int = len(token_need_merge)
            i: int = 0
            merged_token: bytes = max_freq_bytes[0] + max_freq_bytes[1]
            while i < tokens_need_merge_len:
                if (
                    i < tokens_need_merge_len - 1
                    and token_need_merge[i] == max_freq_bytes[0]
                    and token_need_merge[i + 1] == max_freq_bytes[1]
                ):
                    merged_tokens_list.append(merged_token)
                    i += 2
                else:
                    merged_tokens_list.append(token_need_merge[i])
                    i += 1

            merged_tokens_tuple: tuple[bytes] = tuple(merged_tokens_list)

            # Delete merged in set
            token_set_need_merge.remove(token_need_merge)

            # Update max heap

        print(f"tokens_need_merge: {token_set_need_merge}")

        break

    return vocab, merges


def train_bpe(
    input_filepath: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Step1. 预分词
    token_freq: Counter[str] = pretokenization(input_filepath, special_tokens)
    print(f"token_freq: {len(token_freq)}")

    # Step2. 合并
    vocab, merges = merge(
        token_freq,
        vocab_size,
        special_tokens,
    )

    # print(f"{'=' * 10}vocab{'=' * 10}\n{vocab}")
    # print(f"{'=' * 10}merges{'=' * 10}\n{merges}")

    return vocab, merges


def main():
    special_tokens: list[str] = ["<|endoftext|>"]
    # train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10_000, special_tokens)
    train_bpe("data/test-train.txt", 10_000, special_tokens)
    # TestBytePairHeapItem()


if __name__ == "__main__":
    main()
