# Benchmarks

## Environment

- **CPU:** Apple M3 Max (12 cores, ARM64/NEON)
- **RAM:** 36 GB
- **OS:** macOS 15.x (arm64)
- **Rust:** 1.95.0 (stable-aarch64-apple-darwin)
- **turbovec:** v0.2.0 (local build from RyanCodrai/turbovec main branch)
- **FAISS:** 1.13.2 (faiss-cpu via pip)
- **Python:** 3.10+
- **sentence-transformers:** latest (for dataset generation)

## Dataset

- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Vectors:** 100,000 embeddings, dim=384, float32, L2-normalized
- **Text source:** Combinatorial phrases from 40 CS topics × 27 verbs × 16 contexts (e.g. "optimizing machine learning on Kubernetes")
- **Queries:** Last 200 vectors from the dataset (same distribution as database)
- **Ground truth:** Exact inner product over all n vectors (FAISS IndexFlatIP)

### Generate dataset

```bash
python3 -c "
from sentence_transformers import SentenceTransformer
import numpy as np, random, os
random.seed(42)
model = SentenceTransformer('all-MiniLM-L6-v2')
topics = ['machine learning', 'database systems', 'cloud computing', 'NLP',
          'computer vision', 'cybersecurity', 'web development', 'distributed systems',
          'data engineering', 'mobile development', 'blockchain', 'robotics',
          'quantum computing', 'devops', 'AI ethics', 'networking',
          'operating systems', 'software testing', 'game development', 'bioinformatics',
          'recommendation systems', 'search engines', 'compiler design', 'embedded systems',
          'signal processing', 'cryptography', 'parallel computing', 'information retrieval',
          'HCI', 'software architecture', 'API design', 'microservices',
          'containerization', 'serverless', 'edge computing', 'IoT',
          'data visualization', 'time series', 'anomaly detection', 'reinforcement learning']
verbs = ['introduction to', 'advanced', 'practical guide to', 'understanding',
         'best practices for', 'common mistakes in', 'future of', 'scaling',
         'debugging', 'optimizing', 'testing', 'deploying', 'monitoring',
         'comparing', 'building', 'designing', 'implementing', 'evaluating',
         'troubleshooting', 'migrating', 'securing', 'automating', 'benchmarking',
         'profiling', 'refactoring', 'documenting', 'maintaining']
contexts = ['in production', 'for startups', 'at scale', 'for enterprise',
            'with Python', 'with Rust', 'on AWS', 'on Kubernetes',
            'for beginners', 'for experts', 'in 2024', 'with open source',
            'using Docker', 'with CI/CD', 'for real-time systems', 'for batch processing']
sentences = []
while len(sentences) < 100000:
    sentences.append(f'{random.choice(verbs)} {random.choice(topics)} {random.choice(contexts)}')
embeddings = model.encode(sentences[:100000], batch_size=512).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
os.makedirs('../data', exist_ok=True)
np.save('../data/minilm_100k.npy', embeddings)
print(f'Saved: {embeddings.shape}')
"
```

## Commands

### routed-turboquant (Rust)

```bash
# Random data: M/R sweep at 10K
cargo run --release --example bench_v1_tuned

# Real embeddings: 10K + 50K
cargo run --release --example bench_real_data

# Real embeddings: 99K
cargo run --release --example bench_real_99k

# Correctness: full-probe matches flat, partition hit rate
cargo run --release --example diagnose

# Multi-assignment impact on partition hit rate
cargo run --release --example diagnose_multi

# Reranking: recall with/without at various budgets
cargo run --release --example bench_rerank

# Boundary-aware assignment vs fixed M
cargo run --release --example bench_boundary
```

### FAISS baselines (Python)

```bash
python3 benchmarks/faiss_baselines.py
```

## Methodology

- **Warmup:** 5 queries discarded before timing
- **Timing:** Wall-clock per-query (single-threaded)
- **Runs:** Mean of 200 queries (real data) or 100 queries (random data)
- **Recall:** Fraction of true top-10 (from exact search) present in predicted top-10
- **Latency:** Total search time / number of queries (includes routing, scoring, rerank, merge)
- **Build time:** Wall-clock for full index construction (k-means + assignment + TQ encoding)

## Reproducing Results

```bash
# 1. Clone repos
git clone https://github.com/RyanCodrai/turbovec.git
git clone https://github.com/raghavenderreddygrudhanti/routed-turboquant.git

# 2. Generate dataset (requires sentence-transformers)
pip install sentence-transformers numpy faiss-cpu
# Run the dataset generation script above

# 3. Build and run
cd routed-turboquant
cargo test --release
cargo run --release --example bench_real_data
cargo run --release --example bench_real_99k

# 4. FAISS baselines
python3 benchmarks/faiss_baselines.py
```
