import math

import numpy as np
import torch
from torch.utils.data import IterableDataset


def int_to_base_digits(n: int, base: int) -> list:
    if n == 0:
        return [0]
    digits = []
    while n > 0:
        digits.append(n % base)
        n //= base
    return digits[::-1]


def max_seq_len(n_max: int, base: int) -> int:
    return math.ceil(math.log(n_max + 1) / math.log(base)) + 1


def nu_2(n: int) -> int:
    if n == 0:
        return float("inf")
    v = 0
    while n % 2 == 0:
        v += 1
        n //= 2
    return v


class BinarySuffixDataset(IterableDataset):
    """
    Infinite streaming dataset for the binary suffix task.

    Token convention (matching Charton's Int2Int):
      0 = EOS
      1 = BOS
      2..B+1 = digits 0..B-1
      B+2 = PAD
    """

    def __init__(self, base, p, n_max, max_len, seed=0):
        self.base = base
        self.p = p
        self.mod = 2**p
        self.n_max = n_max
        self.max_len = max_len
        self.pad_id = base + 2
        self.bos_id = 1
        self.eos_id = 0
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed
        if worker_info is not None:
            seed += worker_info.id
        rng = np.random.default_rng(seed)

        while True:
            n = int(rng.integers(1, self.n_max + 1))
            digits = int_to_base_digits(n, self.base)
            tokens = [self.bos_id] + [d + 2 for d in digits] + [self.eos_id]
            tokens = tokens[: self.max_len]
            tokens = tokens + [self.pad_id] * (self.max_len - len(tokens))
            label = n % self.mod

            yield (
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(n, dtype=torch.long),
            )


def make_eval_data(base, p, n_max, size, max_len, seed=99):
    rng = np.random.default_rng(seed)
    ns = rng.integers(1, n_max + 1, size=size)
    mod = 2**p
    labels = ns % mod
    pad_id = base + 2

    all_tokens = []
    for n in ns:
        digits = int_to_base_digits(int(n), base)
        tokens = [1] + [d + 2 for d in digits] + [0]
        tokens = tokens[:max_len]
        tokens = tokens + [pad_id] * (max_len - len(tokens))
        all_tokens.append(tokens)

    X = torch.tensor(all_tokens, dtype=torch.long)
    y = torch.tensor(labels.astype(np.int64), dtype=torch.long)
    n_tensor = torch.tensor(ns.astype(np.int64), dtype=torch.long)
    return X, y, n_tensor
