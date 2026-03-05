"""Data encoding for 10-digit addition task.

Encoding: LSB-first (reversed digit order).
Format: [0] rev(a, 10 digits) [0,0] rev(b, 10 digits) [0] -> 11 reversed sum digits
"""


def encode(a: int, b: int) -> list[int]:
    pa = f"{a:010d}"
    pb = f"{b:010d}"
    return (
        [0]
        + [int(c) for c in reversed(pa)]
        + [0, 0]
        + [int(c) for c in reversed(pb)]
        + [0]
    )
