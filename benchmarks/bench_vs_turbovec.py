"""Benchmark: routed-turboquant vs vanilla turbovec.

Measures recall@10, latency, and QPS at various scales.
"""

import time
import numpy as np

try:
    from routed_turboquant import RoutedTurboQuantIndex
except ImportError:
    print("routed-turboquant not installed. Run: maturin develop --release")
    raise

from turbovec import TurboQuantIndex


def generate_data(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random unit vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Brute-force ground truth via float inner product."""
    # (nq, n)
    sims = queries @ vectors.T
    # Top-k indices per query
    gt = np.argsort(-sims, axis=1)[:, :k]
    return gt


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    nq = len(ground_truth)
    hits = 0
    for i in range(nq):
        gt_set = set(ground_truth[i, :k].tolist())
        pred_set = set(predicted[i, :k].tolist())
        hits += len(gt_set & pred_set)
    return hits / (nq * k)


def bench_turbovec_flat(vectors: np.ndarray, queries: np.ndarray, k: int, bit_width: int = 4):
    """Benchmark vanilla turbovec (flat scan)."""
    n, dim = vectors.shape
    nq = len(queries)

    idx = TurboQuantIndex(dim, bit_width)
    idx.add(vectors)

    # Warmup
    idx.search(queries[:1], k)

    # Timed search
    start = time.perf_counter()
    scores, indices = idx.search(queries, k)
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / nq) * 1000
    qps = nq / elapsed

    return np.array(indices), avg_latency_ms, qps


def bench_routed(vectors: np.ndarray, queries: np.ndarray, k: int,
                 n_partitions: int = 128, n_probe: int = 8, bit_width: int = 4):
    """Benchmark routed-turboquant."""
    n, dim = vectors.shape
    nq = len(queries)

    idx = RoutedTurboQuantIndex(dim, n_partitions=n_partitions, n_probe=n_probe,
                                 bit_width=bit_width)

    # Build
    build_start = time.perf_counter()
    idx.build(vectors)
    build_time = time.perf_counter() - build_start

    # Warmup
    idx.search(queries[0], k)

    # Timed search
    start = time.perf_counter()
    all_indices = []
    for i in range(nq):
        _, indices = idx.search(queries[i], k)
        all_indices.append(indices)
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / nq) * 1000
    qps = nq / elapsed

    # Pad results to k
    result = np.zeros((nq, k), dtype=np.int64)
    for i, inds in enumerate(all_indices):
        n_res = min(len(inds), k)
        result[i, :n_res] = inds[:n_res]

    return result, avg_latency_ms, qps, build_time


def run_benchmark(n: int, dim: int, nq: int = 100, k: int = 10):
    """Run full comparison at given scale."""
    print(f"\n{'='*60}")
    print(f"  n={n:,}  dim={dim}  nq={nq}  k={k}")
    print(f"{'='*60}")

    vectors = generate_data(n, dim, seed=42)
    queries = generate_data(nq, dim, seed=123)

    # Ground truth
    print("  Computing ground truth...")
    gt = compute_ground_truth(vectors, queries, k)

    # Vanilla turbovec
    print("  Running turbovec (flat)...")
    tv_indices, tv_latency, tv_qps = bench_turbovec_flat(vectors, queries, k)
    tv_recall = recall_at_k(tv_indices, gt, k)

    # Routed turboquant (various n_probe)
    for n_probe in [4, 8, 16]:
        n_partitions = min(128, n // 10)  # at least 10 vectors per partition
        print(f"  Running routed (P={n_partitions}, R={n_probe})...")
        rt_indices, rt_latency, rt_qps, build_time = bench_routed(
            vectors, queries, k,
            n_partitions=n_partitions, n_probe=n_probe
        )
        rt_recall = recall_at_k(rt_indices, gt, k)
        scan_pct = (n_probe / n_partitions) * 100

        print(f"    routed P={n_partitions} R={n_probe}: "
              f"recall={rt_recall:.3f}  latency={rt_latency:.2f}ms  "
              f"QPS={rt_qps:.0f}  scan={scan_pct:.1f}%  build={build_time:.2f}s")

    print(f"    turbovec flat:          "
          f"recall={tv_recall:.3f}  latency={tv_latency:.2f}ms  "
          f"QPS={tv_qps:.0f}  scan=100%")


if __name__ == "__main__":
    # Small scale test
    run_benchmark(n=10_000, dim=384, nq=100, k=10)

    # Medium scale
    run_benchmark(n=100_000, dim=384, nq=100, k=10)

    # Large scale (where routing should shine)
    run_benchmark(n=500_000, dim=384, nq=50, k=10)
