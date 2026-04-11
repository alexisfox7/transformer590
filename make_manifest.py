"""Generate manifest.txt with one '<base> <p>' line per experiment.

Order: powers of 2 first, then even non-pow2, then odd. Within each base,
iterate p values in order. Earlier rows are easier configs, so SLURM array
tasks 1..N tackle them first and the harder odd-base runs trail.
"""
from data import nu_2

BASES = [2, 4, 8, 10, 12, 24, 28, 3, 5, 13, 17]
P_VALUES = [1, 2, 3, 4, 5, 6, 8]


def base_sort_key(b):
    if b & (b - 1) == 0:  # power of 2
        return (0, -nu_2(b), b)
    if b % 2 == 0:
        return (1, -nu_2(b), b)
    return (2, 0, b)


def main():
    sorted_bases = sorted(BASES, key=base_sort_key)
    rows = [(b, p) for b in sorted_bases for p in P_VALUES]
    with open("manifest.txt", "w") as f:
        for b, p in rows:
            f.write(f"{b} {p}\n")
    print(f"Wrote manifest.txt with {len(rows)} configs")
    print(f"Base order: {sorted_bases}")


if __name__ == "__main__":
    main()
