import os
import regex as re
from multiprocessing import Pool
from collections import defaultdict, Counter
from collections.abc import Iterable
import line_profiler
import pickle

# 必须使用 regex 库，不能使用 import re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

WordType = tuple[bytes, ...]
PairType = tuple[bytes, bytes]


class BPE:
    def __init__(self):
        self.result_vocabulary: dict[int, bytes] = {}
        self.result_merges: list[tuple[bytes, bytes]] = []
        self.special_tokens: list[str] = []  # 新增
        self.pattern: str = PAT  # 新增

    def _process_chunk(self, args):
        # parse args
        input_file_path, start_boundary, end_boundary, special_tokens, regrex = args
        result_local_counter = Counter()
        if end_boundary - start_boundary <= 0:
            return result_local_counter

        # 确保 special_tokens 是 str 类型
        special_tokens = [tok.decode() if isinstance(tok, bytes) else tok for tok in special_tokens]

        compiled_re = re.compile(regrex)
        print(f"Processing file: {input_file_path}, boundary: [{start_boundary}, {end_boundary}]")

        # 打开文件，读取 [start_boundary, end_boundary]
        with open(input_file_path, "rb") as f:
            f.seek(start_boundary)
            chunk = f.read(end_boundary - start_boundary).decode("utf-8", errors="ignore")

            # 剔除 special_tokens 中的字符串
            subchunks = []
            if special_tokens:
                escaped_tokens = [re.escape(tok) for tok in special_tokens]
                split_pattern = "|".join(escaped_tokens)
                subchunks = re.split(split_pattern, chunk)
                # print(f"escaped_tokens: {escaped_tokens}")
            else:
                subchunks = [chunk]

            # 按照 regrex 正则表达式分割子词
            for subchunk in subchunks:
                for match in compiled_re.finditer(subchunk):
                    token = match.group()
                    result_local_counter[token] += 1

        return result_local_counter

    def parallel_pretokenize(
        self,
        input_file_path: str | os.PathLike,
        special_token_strs: list[str],
        regrex: str = PAT,
        num_of_processes: int = 10,
    ) -> Counter[tuple[bytes, ...]]:
        special_tokens = [token.encode("utf-8") for token in special_token_strs]

        with open(input_file_path, "rb") as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0, os.SEEK_SET)
            print(f"file_size: {file_size} bytes")

            # 粗粒度分割边界
            coarse_chunk_size = file_size // num_of_processes
            chunk_boundaries = [i * coarse_chunk_size for i in range(num_of_processes + 1)]
            chunk_boundaries[-1] = file_size
            # print(chunk_boundaries)

            # 细粒度按照 special_tokens 调整边界. 从这一步开始，可以使用并行操作
            mini_chunk_size = min(coarse_chunk_size, 4096)
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
                    earliest_found_at = -1  # set default value to not found
                    for special_token in special_tokens:
                        found_at = mini_chunk.find(special_token)
                        if found_at != -1:
                            # found
                            if earliest_found_at == -1 or found_at < earliest_found_at:
                                earliest_found_at = found_at

                    if earliest_found_at != -1:
                        chunk_boundaries[bi] = initial_position + earliest_found_at
                        break

                    initial_position += mini_chunk_size
            print(f"chunk_boundaries = {chunk_boundaries}")

            # (file, start_boundary, end_boundary, special_tokens, pretoken_regrex)
            args = [
                (
                    input_file_path,  # 进程之间不能共享同一个 inode, 每个进程自己打开文件
                    chunk_boundaries[i],
                    chunk_boundaries[i + 1],
                    special_tokens,
                    regrex,
                )
                for i in range(len(chunk_boundaries) - 1)
            ]

        # 多进程 pre tokenization 处理
        with Pool(processes=num_of_processes) as pool:
            local_counters = pool.map(self._process_chunk, args)

        # print(f"local_counters: {local_counters}")

        global_counter: Counter[tuple[bytes, ...]] = Counter()
        for result_local_counter in local_counters:
            global_counter.update(result_local_counter)

        # print(f"global_counter: {global_counter}")
        print("Done pretokenization.")
        return global_counter

    def _count_pair(
        self, word_freqs: Counter[tuple[bytes, ...]]
    ) -> tuple[Counter[PairType], dict[PairType, set[WordType]]]:
        pair_freqs: Counter[PairType] = Counter()
        pair_to_words: dict[PairType, set[WordType]] = defaultdict(set)

        for word, freq in word_freqs.items():
            # 遍历 n - 1 次，如果 word 只有一个 byte，也就没有 pair 需要统计
            for idx in range(len(word) - 1):
                # Construct pair
                pair = (word[idx], word[idx + 1])

                # Incre pair frequency
                pair_freqs[pair] += freq

                # Mapping pair with word
                pair_to_words[pair].add(word)

        return pair_freqs, pair_to_words

    def _merge_word(self, word: tuple[bytes, ...], pair: tuple[bytes, bytes], new_token: bytes) -> tuple[bytes, ...]:
        result: list[bytes] = []
        i: int = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                result.append(new_token)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)  # list to tuple

    def _get_pair_counts(self, word: tuple[bytes, ...]) -> Counter[PairType]:
        """返回词中每个 pair 出现的次数"""
        cnt = Counter()
        for i in range(len(word) - 1):
            cnt[(word[i], word[i + 1])] += 1
        return cnt

    # @line_profiler.profile
    def _merge_affected_words(
        self,
        word_freqs: Counter[tuple[bytes, ...]],
        pair_freqs: Counter[tuple[bytes, bytes]],
        pair_to_words: dict[PairType, set[WordType]],
        best_pair: tuple[bytes, bytes],
        new_token: bytes,
    ):
        affected_word_list = list(pair_to_words[best_pair])
        for old_word in affected_word_list:
            word_freq = word_freqs[old_word]

            del word_freqs[old_word]

            # Update pair_freqs and pair_to_words
            old_pair_counts: Counter[PairType] = self._get_pair_counts(old_word)
            for pair, count in old_pair_counts.items():
                pair_freqs[pair] -= word_freq * count
                pair_to_words[pair].remove(old_word)
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            # Merge pair in current word
            new_word = self._merge_word(old_word, best_pair, new_token)
            word_freqs[new_word] += word_freq

            # Update pair_freqs and pair_to_words
            new_pair_counts = self._get_pair_counts(new_word)
            for pair, count in new_pair_counts.items():
                pair_freqs[pair] += word_freq * count
                pair_to_words.setdefault(pair, set()).add(new_word)

    def save_vocab_json(self, path: str | os.PathLike) -> None:
        """Save vocabulary to a GPT-2 format vocab.json file."""
        import json

        encoder = Tokenizer._gpt2_byte_encoder()

        vocab_dict: dict[str, int] = {}
        for token_id in sorted(self.result_vocabulary):
            token_bytes: bytes = self.result_vocabulary[token_id]
            token_str = "".join(encoder[b] for b in token_bytes)
            vocab_dict[token_str] = token_id

        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False)
            print(f"Saved vocab.json into {os.path.realpath(path)}")

    def save_merges_txt(self, path: str | os.PathLike) -> None:
        """Save merges to a GPT-2 format merges.txt file."""
        encoder = Tokenizer._gpt2_byte_encoder()

        with open(path, "w", encoding="utf-8") as f:
            for pair in self.result_merges:
                left = "".join(encoder[b] for b in pair[0])
                right = "".join(encoder[b] for b in pair[1])
                f.write(f"{left} {right}\n")
            print(f"Saved merges.txt into {os.path.realpath(path)}")

    def save_to_pickle(self, path: str | os.PathLike) -> None:
        """保存模型到文件"""
        data = {
            "vocab": self.result_vocabulary,
            "merges": self.result_merges,
            "special_tokens": self.special_tokens,
            "pattern": self.pattern,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
            print(f"Saved into {os.path.realpath(path)}")

    def load_from_pickle(self, path: str | os.PathLike) -> None:
        """从文件加载模型"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.result_vocabulary = data["vocab"]
        self.result_merges = data["merges"]
        self.special_tokens = data["special_tokens"]
        self.pattern = data["pattern"]

    def train(
        self,
        input_path: str | os.PathLike,
        special_tokens: list[str],
        vocab_size: int,
        regrex: str = PAT,
        num_of_processes: int = os.cpu_count(),
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert vocab_size >= 256
        self.result_vocabulary: dict[int, bytes] = {}
        self.result_merges: list[tuple[bytes, bytes]] = []
        self.special_tokens = special_tokens
        self.pattern = regrex

        # pre-tokenization
        word_freqs_str: Counter[tuple[str]] = self.parallel_pretokenize(
            input_path, special_tokens, regrex, num_of_processes
        )

        # Convert counter type
        word_freqs: Counter[tuple[bytes, ...]] = Counter()
        for item in word_freqs_str.items():
            token = item[0]
            freq = item[1]
            byte_seq = tuple(bytes([b]) for b in token.encode("utf-8"))
            word_freqs[byte_seq] += freq

        # Append 0 ~ 255
        self.result_vocabulary = {idx: bytes([idx]) for idx in range(256)}

        vocab_idx: int = 256
        # Append special tokens
        for tok in special_tokens:
            self.result_vocabulary[vocab_idx] = tok.encode("utf-8")
            vocab_idx += 1

        # Build initial pair frequencies
        pair_freqs: Counter[tuple[bytes, bytes]] = Counter()
        pair_to_words: dict[PairType, set[WordType]] = defaultdict(set)
        pair_freqs, pair_to_words = self._count_pair(word_freqs)

        # Merge loop
        num_merges = vocab_size - vocab_idx  # Note: put this after init vocabulary
        for _ in range(num_merges):
            if not pair_freqs:
                print(f"No more pair to merge, breaking loop...")
                break
            best_pair = max(pair_freqs.items(), key=lambda item: (item[1], item[0]))[0]  # TODO(tianlan): Opt max speed

            # Append into result
            assert len(best_pair) == 2, "Best merge pair size not equal to 2!"
            new_token: bytes = best_pair[0] + best_pair[1]
            self.result_vocabulary[vocab_idx] = new_token
            vocab_idx += 1
            self.result_merges.append(best_pair)

            # Updating word_freqs by merging all occurences of best_pair
            self._merge_affected_words(word_freqs, pair_freqs, pair_to_words, best_pair, new_token)

        return self.result_vocabulary, self.result_merges


class Tokenizer:
    """
    Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes
    text into integer IDs and decodes integer IDs into text.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pattern: str = PAT,
    ):
        self.vocab = vocab
        self.reverse_vocab = {tok_bytes: tok_id for tok_id, tok_bytes in self.vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.pattern = pattern

        # Build self.special_token_to_id
        self.special_token_to_id: dict[str, int] = {}
        if special_tokens:
            special_set = set(special_tokens)
            for tok_id, tok_bytes in self.vocab.items():
                try:
                    tok_str = tok_bytes.decode("utf-8")
                    if tok_str in special_set:
                        self.special_token_to_id[tok_str] = tok_id
                except UnicodeDecodeError:
                    pass

        # Build self.special_token_split_pattern
        sorted_special_tokens: list[str] = sorted(self.special_tokens, reverse=True, key=len)
        escaped: list[str] = [re.escape(tok) for tok in sorted_special_tokens]
        self.special_token_split_pattern: str = "|".join(escaped)

    @staticmethod
    def _gpt2_byte_encoder() -> dict[int, str]:
        """GPT-2 byte encoder: maps byte value -> unicode string (reverse of byte_decoder)."""
        decoder = Tokenizer._gpt2_byte_decoder()
        return {b: c for c, b in decoder.items()}

    @staticmethod
    def _gpt2_byte_decoder() -> dict[str, int]:
        """Reverse of GPT-2's bytes_to_unicode: maps unicode string -> original byte value."""
        # 把所有 256 个字节都编码成单一、可打印的 Unicode 字符
        byte_values = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        codepoints = byte_values[:]
        offset = 0
        for b in range(2**8):
            if b not in byte_values:
                byte_values.append(b)
                codepoints.append(2**8 + offset)
                offset += 1
        return {chr(c): b for c, b in zip(codepoints, byte_values)}

    @staticmethod
    def from_files(
        vocab_file_path: str,
        merges_file_path: str,
        special_tokens=None,
    ) -> "Tokenizer":
        import json

        byte_decoder = Tokenizer._gpt2_byte_decoder()

        # 1. Load vocab.json
        with open(vocab_file_path, "rb") as f:
            vocab_json_obj = json.load(f)
        result_vocab: dict[int, bytes] = {}
        for vocab_str, token_id in vocab_json_obj.items():
            key: int = token_id
            value: bytes = bytes([byte_decoder[c] for c in vocab_str])
            result_vocab[key] = value

        # 2. Load merges.txt
        result_merges: list[tuple[bytes, bytes]] = []
        with open(merges_file_path, "rb") as f:
            for line in f.readlines():
                strip_line = line.strip()
                if strip_line:
                    split_merge = strip_line.split()
                    assert len(split_merge) == 2, "split merges len != 2"
                    merge_tuple = (
                        bytes([byte_decoder[c] for c in split_merge[0].decode("utf-8")]),
                        bytes([byte_decoder[c] for c in split_merge[1].decode("utf-8")]),
                    )
                    result_merges.append(merge_tuple)

        return Tokenizer(vocab=result_vocab, merges=result_merges, special_tokens=special_tokens)

    def encode(self, input_text: str) -> list[int]:
        if self.special_tokens:
            # 防止 special tokens 被分开
            parts = re.split(f"({self.special_token_split_pattern})", input_text)
            result: list[int] = []
            for part in parts:
                if part in self.special_token_to_id:
                    # 如果是特殊 token，转成 token id
                    result.append(self.special_token_to_id[part])
                elif part:
                    # 非特殊 token
                    result.extend(self._encode_normal(part))
            return result
        return self._encode_normal(input_text)

    def _encode_normal(self, input_text: str) -> list[int]:
        # 1. Pre-tokenize
        pretokenized: list[str] = re.findall(self.pattern, input_text)

        # 2. Apply BPE merges
        merged_bytes: list[bytes] = []
        for pretoken in pretokenized:
            tok_bytes = [bytes([tok]) for tok in pretoken.encode("utf-8")]

            while len(tok_bytes) > 1:
                best_rank = len(self.merges)
                best_idx = None
                for i in range(1, len(tok_bytes)):
                    pair = (tok_bytes[i - 1], tok_bytes[i])
                    rank = self.merge_ranks.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_idx = i

                if best_idx is None:
                    break

                tok_bytes[best_idx] = tok_bytes[best_idx - 1] + tok_bytes[best_idx]
                del tok_bytes[best_idx - 1]

            merged_bytes.extend(tok_bytes)

        # 3. Convert into token id
        result_token_ids: list[int] = []
        for tok_byte in merged_bytes:
            assert self.reverse_vocab.get(tok_byte) != None, f"Cannot found {tok_byte} in vocabulary!"
            result_token_ids.append(self.reverse_vocab.get(tok_byte))

        return result_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        buffer = ""
        for text in iterable:
            buffer += text
            while True:
                matches = list(re.finditer(self.pattern, buffer))
                if not matches:
                    break
                if len(matches) >= 2:
                    cutoff = matches[-2].end()
                    safe = buffer[:cutoff]
                    if not safe:
                        break
                    yield from self.encode(safe)
                    buffer = buffer[cutoff:]
                else:
                    break
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        result_bytes = b"".join(self.vocab[id] for id in ids)
        return result_bytes.decode("utf-8", errors="ignore")


def test_bpe():
    from pathlib import Path

    def gen_output_filename(input_file_path: str) -> str:
        assert input_file_path.endswith(".txt"), f"File '{input_file_path}' must end with '.txt'!"
        return input_file_path.removesuffix(".txt") + ".bpe.model"

    ROOT_DIR = str(Path(__file__).parent.parent)
    TEST_DATA = "data/test.txt"
    TINY_STORIES_VALID_DATA = "data/TinyStoriesV2-GPT4-valid.txt"
    TINY_STORIES_TRAIN_DATA = "data/TinyStoriesV2-GPT4-train.txt"
    OPNE_WEB_TRAIN_DATA = "data/owt_train.txt"
    input_filepath = TINY_STORIES_TRAIN_DATA
    output_filepath = gen_output_filename(input_filepath)

    bpe = BPE()
    bpe.train(input_filepath, ["<|endoftext|>"], 10_000)
    bpe.save_to_pickle(output_filepath)

    print(f"Done bpe.")


def test_tokenizer():
    from pathlib import Path

    # Construct tokenizer
    vocab_json_filepath = "tests/fixtures/gpt2_vocab.json"
    merges_txt_filepath = "tests/fixtures/gpt2_merges.txt"
    tokenizer: Tokenizer = Tokenizer.from_files(vocab_json_filepath, merges_txt_filepath)

    # Use tokenizer to encode (string -> token id)
    # text_str = "she"
    text_str = "🙃"
    encoded_byets = text_str.encode("utf-8")
    decoded_str = encoded_byets.decode("utf-8")

    encode_tokens: list[int] = tokenizer.encode(text_str)

    print(encode_tokens)

    decode = tokenizer.decode(encode_tokens)

    print(decode)


if __name__ == "__main__":
    # test_bpe()

    test_tokenizer()
