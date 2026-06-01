from collections import Counter
import os
import regex as re
import heapq
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


BytesPair = tuple[bytes, bytes]


class BytesPairFreq:
    def __init__(self, bytes_pair_freq: int = 0, locations: set = None):
        self.bytes_pair_freq = bytes_pair_freq
        self.locations = locations if locations is not None else set()


class _RevBytes:
    """反转字节比较：大字节序在堆中视为"更小"，从而实现平局时大 pair 优先。"""

    __slots__ = ("data",)

    def __init__(self, data: bytes):
        self.data = data

    def __lt__(self, other: "_RevBytes") -> bool:
        return self.data > other.data

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _RevBytes) and self.data == other.data

    def __hash__(self) -> int:
        return hash(self.data)


class BytePairHeap:
    """封装 heapq, 用 (-freq, Rev(b1), Rev(b2)) 元组，平局时大 pair 优先。"""

    def __init__(self, bytes_pair_freq: dict[BytesPair, BytesPairFreq]):
        self._heap: list[tuple[int, _RevBytes, _RevBytes, BytesPair]] = []
        self._valid_pairs_set: set[BytesPair] = set(bytes_pair_freq.keys())
        for bp, info in bytes_pair_freq.items():
            heapq.heappush(self._heap, (-info.bytes_pair_freq, _RevBytes(bp[0]), _RevBytes(bp[1]), bp))

    def __bool__(self) -> bool:
        return bool(self._heap)

    def pop_valid(self, bytes_pair_freq: dict[BytesPair, BytesPairFreq]) -> BytesPair | None:
        while self._heap:
            neg_freq: int
            _r1: _RevBytes
            _r2: _RevBytes
            bp: BytesPair
            neg_freq, _r1, _r2, bp = heapq.heappop(self._heap)

            if bp not in self._valid_pairs_set:
                continue

            current: int = bytes_pair_freq[bp].bytes_pair_freq if bp in bytes_pair_freq else 0
            if current != -neg_freq:
                if current > 0:
                    heapq.heappush(self._heap, (-current, _RevBytes(bp[0]), _RevBytes(bp[1]), bp))
                continue

            return bp
        return None

    def discard(self, pair: BytesPair) -> None:
        self._valid_pairs_set.discard(pair)

    def push(self, bp: BytesPair, info: BytesPairFreq) -> None:
        self._valid_pairs_set.add(bp)
        heapq.heappush(self._heap, (-info.bytes_pair_freq, _RevBytes(bp[0]), _RevBytes(bp[1]), bp))


def process_file_chunk(
    input_file_path: str,
    start_boundary: int,
    end_boundary: int,
    split_pattern: str,
    pattern: str,
    special_tokens: list[str],
) -> Counter[str]:
    # Step2. 按照 chunk_boundaries_after 并行处理, 过滤 special tokens + 按照 PAT 分词
    word_freq = Counter()
    print(f"Processing [{start_boundary}, {end_boundary}] ...")

    with open(input_file_path, "rb") as f:
        f.seek(start_boundary)
        chunk_bytes: bytes = f.read(end_boundary - start_boundary)
        chunk_string: str = chunk_bytes.decode("utf-8")  # Bytes to string

        parts = re.split(split_pattern, chunk_string)
        for doc in parts:
            if doc in special_tokens or not doc:
                continue
            for m in re.finditer(pattern, doc):
                word_freq[m.group()] += 1

        return word_freq


def pretokenization(
    input_filepath: str, special_tokens: list[str], desired_num_chunks: int = os.cpu_count()
) -> Counter:

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

    # Step1. 并行正则匹配，先按 special token 切分剔除，再按 PAT 分词
    split_pattern = r"(" + "|".join(re.escape(tok) for tok in special_tokens) + r")"
    start_boundaries = [chunk_boundaries_after[i] for i in range(len(chunk_boundaries_after) - 1)]
    end_boundaries = [chunk_boundaries_after[i] for i in range(1, len(chunk_boundaries_after))]

    args = [
        (input_filepath, start_boundary, end_boundary, split_pattern, PAT, special_tokens)
        for start_boundary, end_boundary in zip(start_boundaries, end_boundaries)
    ]
    with Pool(desired_num_chunks) as pool:
        results: list[Counter] = pool.starmap(process_file_chunk, args)

    word_freq = Counter()
    for result in results:
        word_freq.update(result)
    return word_freq


def merge_word(
    word_before_merge: tuple[bytes, ...], max_freq_bytes: BytesPair, merged_token: bytes
) -> tuple[bytes, ...]:
    word_after_merge_list: list[bytes] = []
    i: int = 0
    n: int = len(word_before_merge)
    while i < n:
        if i < n - 1 and word_before_merge[i] == max_freq_bytes[0] and word_before_merge[i + 1] == max_freq_bytes[1]:
            word_after_merge_list.append(merged_token)
            i += 2
        else:
            word_after_merge_list.append(word_before_merge[i])
            i += 1
    return tuple(word_after_merge_list)


def merge(
    token_freq_str: Counter[str], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256, f"Require {vocab_size} >= 256"
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # Counter[str] -> Counter[bytes]
    word_freq: dict[tuple[bytes, ...], int] = {}
    bytes_pair_freq: dict[BytesPair, BytesPairFreq] = defaultdict(BytesPairFreq)

    for tok_str, tok_freq in token_freq_str.items():
        tok_bytes: bytes = tok_str.encode("utf-8")
        # 坑点: python 里 bytes 对象 tok_bytes[i] 返回的类型，不是 bytes，而是 int，所以需要转成 bytes。转的时候需要 bytes([int]) 而不是 bytes(int)。
        # 从效率上来说，使用切片，会比 bytes([int]) 构造 bytes 效率更好，所以尽量使用切片
        tok_bytes_tuple: tuple[bytes, ...] = tuple([tok_bytes[i : i + 1] for i in range(len(tok_bytes))])
        word_freq[tok_bytes_tuple] = word_freq.get(tok_bytes_tuple, 0) + tok_freq

        for i in range(1, len(tok_bytes)):
            bytes_pair: BytesPair = (
                tok_bytes[i - 1 : i],
                tok_bytes[i : i + 1],
            )  # 构造 bytes, 上面同理
            bytes_pair_freq[bytes_pair].bytes_pair_freq += tok_freq
            bytes_pair_freq[bytes_pair].locations.add(tok_bytes_tuple)  # 存引用，不拷贝

    # 构造大顶堆
    heap: BytePairHeap = BytePairHeap(bytes_pair_freq)

    # 构造 vocabulary, 加入 256 (0 ~ 255) 个字节
    for num in range(256):
        num_byte: bytes = bytes([num])  # Convert int number into bytes
        vocab[num] = num_byte

    # 构造 vocabulary, 加入special_tokens
    vocab_token_id: int = 256
    for tok_bytes in [tok.encode("utf-8") for tok in special_tokens]:
        vocab[vocab_token_id] = tok_bytes
        vocab_token_id += 1

    with tqdm(total=vocab_size, desc="Computing vocabulary", unit="merge", initial=len(vocab)) as pbar:
        while vocab_token_id < vocab_size and heap:
            max_freq_bytes: BytesPair = heap.pop_valid(bytes_pair_freq)
            if max_freq_bytes is None:
                break

            merged_token: bytes = b"".join(max_freq_bytes)
            vocab[vocab_token_id] = merged_token
            vocab_token_id += 1
            merges.append(max_freq_bytes)
            pbar.update(1)

            word_set_need_merge: tuple[bytes, ...] = bytes_pair_freq[max_freq_bytes].locations

            for word_before_merge in word_set_need_merge.copy():
                freq = word_freq.get(word_before_merge, 0)
                if freq == 0:
                    continue

                # 生成新 token（替换所有连续出现的 max_freq_bytes）
                word_after_merge: tuple[bytes, ...] = merge_word(word_before_merge, max_freq_bytes, merged_token)

                del word_freq[word_before_merge]

                # 1. 旧 token 中的所有相邻对频率 -freq
                for idx in range(len(word_before_merge) - 1):
                    old_pair = (word_before_merge[idx], word_before_merge[idx + 1])
                    info = bytes_pair_freq[old_pair]
                    info.bytes_pair_freq -= freq
                    info.locations.discard(word_before_merge)

                    if info.bytes_pair_freq <= 0:
                        del bytes_pair_freq[old_pair]
                        heap.discard(old_pair)  # 标记无效
                    else:
                        heap.push(old_pair, info)  # 推入新的（降低后的）频率

                # 2. 新 token 的添加与相邻对频率 +freq
                word_freq[word_after_merge] = word_freq.get(word_after_merge, 0) + freq
                for idx in range(len(word_after_merge) - 1):
                    new_pair = (word_after_merge[idx], word_after_merge[idx + 1])
                    info = bytes_pair_freq[new_pair]  # defaultdict 自动创建
                    info.bytes_pair_freq += freq
                    info.locations.add(word_after_merge)
                    heap.push(new_pair, info)  # 推入新频率（新增或增加）

    return vocab, merges


def train_bpe(
    input_filepath: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Step1. 预分词
    bytes_pair_freq: Counter[str] = pretokenization(input_filepath, special_tokens)
    print("Done pretokenization!")

    # Step2. 合并
    print("Start merge ...")
    vocab, merges = merge(
        bytes_pair_freq,
        vocab_size,
        special_tokens,
    )
    print("Done merge!")

    return vocab, merges


def test_regex(special_tokens: list[str]):
    # 正则里有特殊含义的字符有：. ^ $ * + ? { } [ ] \ | ( ) — 一共 14 个。
    # < 和 > 就是普通字符，不需要转义。而 | 是这 14 个之一（表示"或"），所以 re.escape 会把它变成 \|。
    split_pattern = r"(" + "|".join(re.escape(tok) for tok in special_tokens) + r")"
    print(f"split_pattern: {split_pattern}")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 standard byte-to-unicode mapping."""
    bs: list[int] = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs: list[str] = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def dump_bpe_train_result(output_dir: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    import json

    os.makedirs(output_dir, exist_ok=True)
    byte_encoder: dict[int, str] = bytes_to_unicode()

    # Step 1: Dump vocab.json
    reversed_vocab = {}
    for tok_id, tok_bytes in vocab.items():
        # Map each byte through the byte_encoder
        tok_str = "".join(byte_encoder[b] for b in tok_bytes)
        reversed_vocab[tok_str] = tok_id

    with open(os.path.join(output_dir, "vocab.json"), mode="w", encoding="utf-8") as json_file:
        json.dump(reversed_vocab, json_file, indent=2, ensure_ascii=False)

    # Step 2: Dump merges.txt
    with open(os.path.join(output_dir, "merges.txt"), mode="w", encoding="utf-8") as txt_file:
        txt_file.write("#version: 0.2\n")  # Standard header
        for first_bytes, second_bytes in merges:
            first_str = "".join(byte_encoder[b] for b in first_bytes)
            second_str = "".join(byte_encoder[b] for b in second_bytes)
            txt_file.write(f"{first_str} {second_str}\n")


def load_bpe_train_result(
    vocab_json_filepath: str, merges_txt_filepath: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    import json

    byte_encoder: dict[int, str] = bytes_to_unicode()
    str_decoder: dict[str, int] = {byte_str: byte_int for byte_int, byte_str in byte_encoder.items()}

    # Decode vocab.json
    vocab_str: dict[str, int] = {}
    with open(vocab_json_filepath, mode="r", encoding="utf-8") as json_file:
        vocab_str = json.load(json_file)
    reversed_vocab: dict[bytes, int] = {}
    for tok_str, tok_id in vocab_str.items():
        token_bytes_tuple = tuple([bytes([str_decoder[tok_char]]) for tok_char in tok_str])
        # print(f"decode token_bytes: {token_bytes}")
        reversed_vocab[token_bytes_tuple] = tok_id
    vocab: dict[int, bytes] = {tok_id: token_bytes for token_bytes, tok_id in reversed_vocab.items()}

    # Decode merges.txt
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_txt_filepath, mode="r", encoding="utf-8") as txt_file:
        while True:
            oneline: str = txt_file.readline()
            if len(oneline) == 0:
                break
            if oneline.startswith("#"):
                continue

            token_strs: list[str] = oneline.rstrip().split(" ")
            # assert len(token_strs) == 2
            first_token_bytes = tuple([bytes([str_decoder[tok_char]]) for tok_char in token_strs[0]])
            second_token_bytes = tuple([bytes([str_decoder[tok_char]]) for tok_char in token_strs[1]])
            merges.append((first_token_bytes, second_token_bytes))

    return vocab, merges


def test_bpe():
    special_tokens: list[str] = ["<|endoftext|>"]
    # vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10_000, special_tokens)
    vocab, merges = train_bpe("data/test-train.txt", 10_000, special_tokens)
    dump_bpe_train_result(output_dir="data/", vocab=vocab, merges=merges)
    vocab, merges = load_bpe_train_result("data/vocab.json", "data/merges.txt")

    # print(f"vocab: {vocab}")
    # print(f"merges: {merges}")

    # print(
    #     f"{list(range(ord('!'), ord('~') + 1))}\n{list(range(ord('¡'), ord('¬') + 1))}\n{list(range(ord('®'), ord('ÿ') + 1))}"
    # )


def main():
    # TestBytePairHeapItem()
    # test_regex(special_tokens)
    test_bpe()


if __name__ == "__main__":
    main()
