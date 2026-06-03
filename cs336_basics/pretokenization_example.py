import os
import regex as re
from typing import BinaryIO
from collections import Counter


from collections import defaultdict, Counter


class BPE:
    @classmethod
    def train(self, pre_token_result: Counter):  # {str, int}
        # The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes)
        result_vocab: dict[int, bytes] = {}
        result_vocab_set = set()

        # A list of BPE result_merges produced from training. Each list
        # item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
        # <token2>. The result_merges should be ordered by order of creation.
        result_merges: list[tuple[bytes, bytes]] = []

        # Step0. 打散 pre_token_result
        # pre_token_result -> token_tuple_conter
        pre_token_result_in_tuple: {tuple, int} = {
            tuple(item[0]): item[1] for item in pre_token_result.items()
        }

        counter: int = 0

        # Step1. 统计相邻字符对的频率，初始化 result_vocab_set
        pair_freq_dict: {tuple, int} = {}
        for item in pre_token_result_in_tuple.items():
            sub_token = item[0]
            freq = item[1]

            for ch in sub_token:  # 添加每个字符
                result_vocab_set.add(ch)

            for i in range(0, len(sub_token) - 1):  # [0, len(item) - 2]
                tmp_pair = (sub_token[i], sub_token[i + 1])
                pair_freq_dict[tmp_pair] = pair_freq_dict.get(tmp_pair, 0) + freq

                if i == 0:
                    result_vocab_set.add(sub_token[0])
                    result_vocab_set.add(sub_token[1])
                else:
                    result_vocab_set.add(sub_token[i + 1])

        # 添加到 set 去重，然后赋值 token id
        for item in result_vocab_set:
            result_vocab[counter] = str(item).encode("utf-8")
            counter += 1

        while True:
            freq_values = pair_freq_dict.values()
            max_freq = max(freq_values)

            ######## 跳出循环 ########
            if max_freq <= 1:
                break
            #########################

            max_freq_tuples = []
            for key, value in pair_freq_dict.items():
                if value == max_freq:
                    max_freq_tuples.append(key)
            max_freq_tuples.sort()
            pair_to_be_merged = max(max_freq_tuples)
            assert len(pair_to_be_merged) == 2, "pair_to_be_merged size != 2"

            # 添加到合并历史
            result_merges.append(
                (
                    str(pair_to_be_merged[0]).encode("utf-8"),
                    str(pair_to_be_merged[1]).encode("utf-8"),
                )
            )

            # 添加到词汇表
            merged_pair = "".join(pair_to_be_merged)
            result_vocab[counter] = merged_pair.encode("utf-8")
            counter += 1

            # Step3. 合并 pre_token_result_in_tuple，同时统计频率变化
            merge_token_in_tuple: dict[tuple, int] = {}
            freq_delta = defaultdict(int)  # 记录频率变化量

            for item in pre_token_result_in_tuple.items():
                token_tuple = item[0]
                freq = item[1]

                # 1) 减去旧序列中所有相邻对的频率
                for i in range(len(token_tuple) - 1):
                    old_pair = (token_tuple[i], token_tuple[i + 1])
                    freq_delta[old_pair] -= freq

                # 2) 执行合并，生成新序列
                merged_tokens_list = []
                i = 0
                while i < len(token_tuple):
                    if (
                        i < len(token_tuple) - 1
                        and token_tuple[i] == pair_to_be_merged[0]
                        and token_tuple[i + 1] == pair_to_be_merged[1]
                    ):
                        merged_tokens_list.append(token_tuple[i] + token_tuple[i + 1])
                        i += 2
                    else:
                        merged_tokens_list.append(token_tuple[i])
                        i += 1
                new_tuple = tuple(merged_tokens_list)
                merge_token_in_tuple[new_tuple] = freq

                # 3) 加上新序列中所有相邻对的频率
                for i in range(len(new_tuple) - 1):
                    new_pair = (new_tuple[i], new_tuple[i + 1])
                    freq_delta[new_pair] += freq

            # 应用频率变化到 pair_freq_dict
            for pair, delta in freq_delta.items():
                new_freq = pair_freq_dict.get(pair, 0) + delta
                if new_freq <= 0:
                    if pair in pair_freq_dict:
                        del pair_freq_dict[pair]
                else:
                    pair_freq_dict[pair] = new_freq

            # Step4. 覆盖原始字典
            pre_token_result_in_tuple = merge_token_in_tuple

        return result_vocab, result_merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0, os.SEEK_SET)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


TRAIN_DATA = "TinyStoriesV2-GPT4-train.txt"
VALIDATION_DATA = "TinyStoriesV2-GPT4-valid.txt"
DEMO_DATA = "test.txt"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# data dir
cur_dir = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.abspath(os.path.join(cur_dir, "../data/", DEMO_DATA))

# sub-token counter
global_token_counter = Counter()


## Usage
with open(train_data_path, "rb") as f:
    num_processes = 4

    # Step1. pre-tokenization
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # Run pre-tokenization on your chunk and store the counts for each pre-token

        # 按 <|endoftext|> 分割并去除空块
        sub_chunks = [
            chunk.strip() for chunk in chunk.split("<|endoftext|>") if chunk.strip()
        ]

        for i, sub_chunk in enumerate(sub_chunks):
            # print(f"============Chunk {i}============:\n{sub_chunk}")
            for match in re.finditer(PAT, sub_chunk):
                token = match.group()
                global_token_counter[token] += 1

    # Step2. run bpe algorithm merge
    vocab, merges = BPE().train(global_token_counter)


# After pre-tokenization
# print(global_token_counter.most_common(10))
