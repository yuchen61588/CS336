from __future__ import annotations


import pathlib
from functools import lru_cache

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"
TINY_PATH = (pathlib.Path(__file__).parent.resolve().parent)/"data"

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    返回一个从每个可能的字节（0到255的整数）到可打印Unicode字符串字符的映射。
    此函数取自GPT-2代码。

    例如，`chr(0)`是`\x00`，这是一个不可打印字符：

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    因此，此函数返回一个字典`d`，其中`d[0]`返回`Ā`。
视觉上可打印的字节保持其原始字符串表示[1]。
例如，`chr(33)`返回`!`，因此相应地`d[33]`返回`!`。
特别要注意的是，空格字符`chr(32)`变成`d[32]`，它返回'Ġ'。

对于不可打印的字符，该函数将表示该字符的Unicode码点的整数
（由Python的`ord`函数返回）偏移256。例如，`ord(" ")`返回`32`，
因此空格字符' '被偏移到`256 + 32`。由于`chr(256 + 32)`返回`Ġ`，
我们将其用作空格的字符串表示。

此函数可以简化BPE实现，并使在序列化到文件后手动检查生成的合并结果变得稍微容易一些。
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
