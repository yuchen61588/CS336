## 问题

### 第二问

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.  Deliverable: A one-to-two sentence response. 

我们更倾向于 UTF-8，因为它对于以 ASCII 为主的文本更节省内存（每个字符使用 1 个字节，而不是 2 或 4 个），避免了字节序问题，并且允许我们将输入建模为来自 256 个固定词表的原始字节序列。

 (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.  

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):

 return "".join([bytes([b]).decode("utf-8") for b in bytestring])  >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")) 'hello'  

**例输入：** `b'\xe7\x89\x9b'`（字符“牛”的 UTF-8 字节）。**解释：** 该函数不正确，因为它试图单独解码每个字节，但多字节 UTF-8 字符（如“牛”）使用必须一起解码的字节序列来表示单个字符，拆分它们会导致错误

Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect. 

 (c) Give a two byte sequence that does not decode to any Unicode character(s).  Deliverable: An example, with a one-sentence explanation.

不符合上面字节的都一样。