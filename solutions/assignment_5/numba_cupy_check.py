#!/usr/bin/env python3
import time


def check_numba():
    try:
        from numba import cuda
        ok = cuda.is_available()
        print(f"Numba CUDA available: {ok}")
        if ok:
            print(f"Numba current device: {cuda.get_current_device().name}")
    except Exception as e:
        print(f"Numba check failed: {e}")


def check_cupy():
    try:
        import cupy as cp
        t0 = time.perf_counter()
        a = cp.arange(1_000_000, dtype=cp.float32)
        b = cp.sqrt(a).sum()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        print(f"CuPy available: True, sample result={float(b):.2f}, time={t1 - t0:.6f}s")
    except Exception as e:
        print(f"CuPy check failed: {e}")


if __name__ == "__main__":
    check_numba()
    check_cupy()
