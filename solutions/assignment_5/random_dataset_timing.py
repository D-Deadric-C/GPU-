#!/usr/bin/env python3
import argparse
import os
import time
import random


def lcg(seed, n, a=1664525, c=1013904223, m=2**32):
    x = seed
    out = []
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x / m)
    return out


def xorshift32(seed, n):
    x = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        out.append((x & 0xFFFFFFFF) / 0xFFFFFFFF)
    return out


def write_dataset(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate random datasets with timing.")
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--outdir", default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.perf_counter()
    py_vals = [random.Random(args.seed).random() for _ in range(args.n)]
    write_dataset(os.path.join(args.outdir, "python_random.txt"), py_vals)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    lcg_vals = lcg(args.seed, args.n)
    write_dataset(os.path.join(args.outdir, "lcg_random.txt"), lcg_vals)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    xor_vals = xorshift32(args.seed, args.n)
    write_dataset(os.path.join(args.outdir, "xorshift_random.txt"), xor_vals)
    t5 = time.perf_counter()

    print(f"Generated {args.n} values each")
    print(f"Python RNG time:   {t1 - t0:.6f} s")
    print(f"LCG RNG time:      {t3 - t2:.6f} s")
    print(f"Xorshift RNG time: {t5 - t4:.6f} s")


if __name__ == "__main__":
    main()
