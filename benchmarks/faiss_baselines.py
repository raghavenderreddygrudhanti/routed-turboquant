"""FAISS baselines on real sentence-transformer embeddings.

Compares: Flat (exact), HNSW, IVF-Flat, IVFPQ
Same dataset as routed-turboquant benchmarks.
"""

import numpy as np
import faiss
import time

# Load data
data = np.load("../data/minilm_100k.npy")
print(f"Loaded: {data.shape[0]} vectors, dim={data.shape[1]}")

dim = data.shape[1]
k = 10
nq = 200

for n in [10_000, 50_000, 99_000]:
    vectors = data[:n].copy()
    queries = vectors[n - nq:n].copy()

    print(f"\n{'='*70}")
    print(f"  n={n}, dim={dim}, k={k}, nq={nq}")
    print(f"{'='*70}")

    # Exact ground truth
    index_flat = faiss.IndexFlatIP(dim)
    index_flat.add(vectors)
    gt_scores, gt_ids = index_flat.search(queries, k)

    def recall_at_k(pred_ids, gt_ids, k):
        hits = 0
        for i in range(len(pred_ids)):
            hits += len(set(pred_ids[i, :k].tolist()) & set(gt_ids[i, :k].tolist()))
        return hits / (len(pred_ids) * k)

    # --- FAISS Flat (exact) ---
    t0 = time.perf_counter()
    scores, ids = index_flat.search(queries, k)
    flat_time = (time.perf_counter() - t0) * 1000 / nq
    flat_recall = recall_at_k(ids, gt_ids, k)
    print(f"  FAISS Flat (exact):    recall={flat_recall:.3f}  latency={flat_time:.3f}ms")

    # --- FAISS HNSW ---
    index_hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = 200
    index_hnsw.hnsw.efSearch = 64
    index_hnsw.add(vectors)

    # warmup
    index_hnsw.search(queries[:5], k)

    t0 = time.perf_counter()
    scores, ids = index_hnsw.search(queries, k)
    hnsw_time = (time.perf_counter() - t0) * 1000 / nq
    hnsw_recall = recall_at_k(ids, gt_ids, k)
    print(f"  FAISS HNSW (M=32 ef=64): recall={hnsw_recall:.3f}  latency={hnsw_time:.3f}ms")

    # --- FAISS IVF-Flat ---
    nlist = 64 if n <= 50_000 else 128
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(vectors)
    index_ivf.add(vectors)
    index_ivf.nprobe = 16

    # warmup
    index_ivf.search(queries[:5], k)

    t0 = time.perf_counter()
    scores, ids = index_ivf.search(queries, k)
    ivf_time = (time.perf_counter() - t0) * 1000 / nq
    ivf_recall = recall_at_k(ids, gt_ids, k)
    print(f"  FAISS IVF-Flat (nlist={nlist} nprobe=16): recall={ivf_recall:.3f}  latency={ivf_time:.3f}ms")

    # IVF-Flat with higher nprobe
    index_ivf.nprobe = 32
    t0 = time.perf_counter()
    scores, ids = index_ivf.search(queries, k)
    ivf_time2 = (time.perf_counter() - t0) * 1000 / nq
    ivf_recall2 = recall_at_k(ids, gt_ids, k)
    print(f"  FAISS IVF-Flat (nlist={nlist} nprobe=32): recall={ivf_recall2:.3f}  latency={ivf_time2:.3f}ms")

    # --- FAISS IVFPQ ---
    m_pq = 48  # subquantizers (dim must be divisible by m)
    nbits = 8
    quantizer_pq = faiss.IndexFlatIP(dim)
    index_ivfpq = faiss.IndexIVFPQ(quantizer_pq, dim, nlist, m_pq, nbits, faiss.METRIC_INNER_PRODUCT)
    index_ivfpq.train(vectors)
    index_ivfpq.add(vectors)
    index_ivfpq.nprobe = 16

    # warmup
    index_ivfpq.search(queries[:5], k)

    t0 = time.perf_counter()
    scores, ids = index_ivfpq.search(queries, k)
    ivfpq_time = (time.perf_counter() - t0) * 1000 / nq
    ivfpq_recall = recall_at_k(ids, gt_ids, k)
    print(f"  FAISS IVFPQ (nlist={nlist} m={m_pq} nprobe=16): recall={ivfpq_recall:.3f}  latency={ivfpq_time:.3f}ms")

    # --- Summary ---
    print(f"\n  --- Summary (n={n}) ---")
    print(f"  {'Method':<40} {'Recall@10':<12} {'Latency':<12}")
    print(f"  {'-'*64}")
    print(f"  {'FAISS Flat (exact)':<40} {flat_recall:<12.3f} {flat_time:<12.3f}ms")
    print(f"  {'FAISS HNSW (M=32 ef=64)':<40} {hnsw_recall:<12.3f} {hnsw_time:<12.3f}ms")
    print(f"  {'FAISS IVF-Flat (nprobe=16)':<40} {ivf_recall:<12.3f} {ivf_time:<12.3f}ms")
    print(f"  {'FAISS IVF-Flat (nprobe=32)':<40} {ivf_recall2:<12.3f} {ivf_time2:<12.3f}ms")
    print(f"  {'FAISS IVFPQ (nprobe=16)':<40} {ivfpq_recall:<12.3f} {ivfpq_time:<12.3f}ms")
    print(f"  {'turbovec flat (from Rust bench)':<40} {'see README':<12} {'see README':<12}")
    print(f"  {'routed-turboquant (from Rust bench)':<40} {'see README':<12} {'see README':<12}")
