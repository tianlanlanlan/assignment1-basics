# For problem1
niu_decimal = ord("牛")
print(f"niu_decimal = {niu_decimal}")

niu_char = chr(niu_decimal)
print(f"niu_char = {niu_char}")

print(f"'{chr(0)}'")

# For problem2
str1 = "Hello World! 你好啊!"
str1_utf8 = str1.encode("utf-8")
print(f"str1 = {str1}, len = {len(str1)}")
print(f"str1_utf8: {str1_utf8}, len = {len(str1_utf8)}")
print(f"str1_utf8 decode: {str1_utf8.decode("utf-8")}")
# str1_utf16 = str1.encode("utf-16")
# print(f"str1_utf16: {str1_utf16}, len = {len(str1_utf16)}")
# str1_utf32 = str1.encode("utf-32")
# print(f"str1_utf32: {str1_utf32}, len = {len(str1_utf32)}")

encode_ni = "你".encode("utf-8")
print(f"encode '你' with utf-8: {encode_ni}")
for b in list(encode_ni):
    print(f"{b:08b}")


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    decode_list = []
    for b in bytestring:
        print(f"b = {b}, {[b]}, {bytes([b])}")
        decode_list.append(bytes([b]).decode("utf-8"))
        # print(f"bytes({b}) = {bytes(b)}")
    print(f"decode_list = {decode_list}, after join: {"".join(decode_list)}")
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


wrong_str = decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
# wrong_str = decode_utf8_bytes_to_str_wrong("hello 你好".encode("utf-8"))
print(f"{wrong_str}")

wrong_decode_bytes = b"\xc0\x80"
# wrong_decoded_string = wrong_decode_bytes.decode("utf-8") # This will cause error: invalid start byte
# print(f"wrong_decoded_string: {wrong_decoded_string}")

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
findall_result = re.findall(PAT, "some text that i'll pre-tokenize")
print(f"findall_result: {findall_result}")

finditer_result = re.finditer(PAT, "some text that i'll pre-tokenize")
for match in finditer_result:
    token = match.group()
    token_utf8 = token.encode("utf-8")
    print(f"group: '{match.group()}', span: '{match.span()}', utf-8: {token_utf8}")

"""
python 字符串使用 unicode 编码, 存储的是unicode码点的二进制格式, 以 "A中😊" 字符串举例 (假设小端序，地址从低到高):
'A'   : 0x41 0x00 0x00 0x00   (U+0041 0x0041, utf-8 对应 e4 b8 ad)
'中'  : 0x2D 0x4E 0x00 0x00   (U+4E2D 对应 0x4E2D)
'😊'  : 0x0A 0xF6 0x01 0x00   (U+1F60A 对应 0x0001F60A)
"""
