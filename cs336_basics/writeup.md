## Problem1 (unicode1):

1. What Unicode character does chr(0) return

```
空字符: ''
```

2. How does this character’s string representation (__repr__()) differ from its printed 
representation?
```
'\x00'
'\0' 在 c 语言标志着字符串结尾，代表空字符
```

3. What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

```python
>>> "this is a test" + chr(0) + "string" 输出：
'this is \x00string'

>>> print("this is a test" + chr(0) + "string") 输出：
this isstring

直接字符串回车，似乎是打印了 temp_str.__repr__() 
看起来 python print 函数打印字符串，不是遇到 '\00' 就结束了
```
## Problem2 (unicode2):
1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
```
utf-8 最小编码单位是一个字节，因此基础词表最少，2 ^ 8 = 256 个；二是向下兼容 ascii 
```

2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string
into a Unicode string. Why is this function incorrect? Provide an example of an input byte
string that yields incorrect results.
```
1. 异常字符串举例: "hello 你好", 包含中文字符就会报错
2. utf-8 是变长编码，英文字符消耗一个字节存储，中文字符是2个字节以上编码。所以不能逐个 decode，对于中文字符需要按照原始 encode 后的字节数量来 decode
```

3. Give a two-byte sequence that does not decode to any Unicode character(s).
```
# 举例 0xc080
bytes_array: bytes = b"\xc0\x80"
bytes_array.decode("utf-8")
```

> 注意 pdf 里面没强调一个很重要的点：Unicode 和 Unicode Encoding，不是一个东西

Problem (train_bpe_tinystories):  BPE Training on TinyStories (2 points)
(a) Elapsed time:
Executed in   23.27 mins    fish           external
   usr time   30.11 mins    0.00 millis   30.11 mins
   sys time    0.25 mins    1.22 millis    0.25 mins