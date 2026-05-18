//! Benchmark on real sentence-transformer embeddings (all-MiniLM-L6-v2, 384d).
//!
//! Loads pre-generated embeddings from bitcache/benchmarks/data/minilm_100k.npy
//! and compares routed-turboquant vs turbovec flat on real semantic data.

extern crate blas_src;

use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use turbovec::TurboQuantIndex;

/// Load a .npy file containing f32 array of shape (n, dim).
fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("cannot open npy file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("cannot read npy file");

    // Parse numpy header
    // Magic: \x93NUMPY
    assert_eq!(&buf[0..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = std::str::from_utf8(&buf[10..10 + header_len]).unwrap();

    // Extract shape from header string like "{'descr': '<f4', 'fortran_order': False, 'shape': (100000, 384), }"
    let shape_start = header.find("'shape': (").unwrap() + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header[shape_start..shape_end];
    let dims: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().unwrap())
        .collect();
    let n = dims[0];
    let dim = dims[1];

    // Data starts after header
    let data_start = 10 + header_len;
    let data_bytes = &buf[data_start..];
    let floats: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    assert_eq!(floats.len(), n * dim);
    (floats, n, dim)
}

fn exact_topk(vectors: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let n = vectors.len() / dim;
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            (
                v.iter().zip(query.iter()).map(|(a, b)| a * b).sum::<f32>(),
                i,
            )
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scores.iter().take(k).map(|(_, i)| *i).collect()
}

fn recall_score(predicted: &[usize], ground_truth: &[usize], k: usize) -> f64 {
    let pred_set: HashSet<usize> = predicted.iter().take(k).copied().collect();
    let gt_set: HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    pred_set.intersection(&gt_set).count() as f64 / k as f64
}

fn main() {
    let path = "../data/minilm_100k.npy";
    println!("Loading real embeddings from {}...", path);
    let (all_vectors, total_n, dim) = load_npy(path);
    println!("Loaded: {} vectors, dim={}\n", total_n, dim);

    let k = 10;
    let nq = 200;

    // Test at multiple scales using subsets of the real data
    for &n in &[10_000, 50_000, 99_000] {
        if n > total_n {
            continue;
        }

        println!("================================================================");
        println!("  REAL DATA: n={}, dim={}, k={}, nq={}", n, dim, k, nq);
        println!("================================================================");

        let vectors = &all_vectors[..n * dim];
        // Use last nq vectors as queries (they're from the same distribution)
        let query_start = n - nq;
        let queries = &all_vectors[query_start * dim..n * dim];

        // Exact ground truth
        println!("  Computing exact ground truth...");
        let exact_gt: Vec<Vec<usize>> = (0..nq)
            .map(|i| exact_topk(vectors, &queries[i * dim..(i + 1) * dim], dim, k))
            .collect();

        // turbovec flat
        println!("  Building turbovec flat...");
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(vectors);
        flat_idx.prepare();
        flat_idx.search(&queries[0..dim], k); // warmup

        let start = Instant::now();
        let flat_results = flat_idx.search(queries, k);
        let flat_latency = start.elapsed().as_secs_f64() * 1000.0 / nq as f64;

        let mut flat_recall_sum = 0.0;
        for i in 0..nq {
            let pred: Vec<usize> = flat_results
                .indices_for_query(i)
                .iter()
                .filter(|&&x| x >= 0)
                .map(|&x| x as usize)
                .collect();
            flat_recall_sum += recall_score(&pred, &exact_gt[i], k);
        }
        let flat_recall = flat_recall_sum / nq as f64;
        println!(
            "  turbovec flat:     recall={:.3}  latency={:.3}ms\n",
            flat_recall, flat_latency
        );

        // Routed configs
        let p = if n <= 10_000 {
            32
        } else if n <= 50_000 {
            64
        } else {
            128
        };

        println!(
            "  {:<22} {:<8} {:<10} {:<10} {:<10}",
            "Config", "Recall", "Latency", "vs Flat", "Speedup"
        );
        println!("  {}", "-".repeat(60));

        let configs: Vec<(usize, usize, &str)> = vec![
            (1, p / 2, "M=1 R=P/2"),
            (1, p * 3 / 4, "M=1 R=3P/4"),
            (2, p / 4, "M=2 R=P/4"),
            (2, p / 2, "M=2 R=P/2"),
            (3, p / 4, "M=3 R=P/4"),
            (4, p / 4, "M=4 R=P/4"),
            (4, p / 3, "M=4 R=P/3"),
            (4, p / 2, "M=4 R=P/2"),
        ];

        for (m, r, label) in &configs {
            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: *r,
                bit_width: 4,
                kmeans_iter: 15,
                seed: 42,
                multi_assign: *m,
                boundary_threshold: None,
                max_assign: 4,
                rerank_top: 25,
            };

            let build_start = Instant::now();
            let idx = RoutedTurboQuantIndex::build(vectors, config);
            let _build_s = build_start.elapsed().as_secs_f64();

            // Warmup
            idx.search(&queries[0..dim], k);

            // Timed search
            let start = Instant::now();
            let mut recall_sum = 0.0;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let (_, ids) = idx.search(q, k);
                recall_sum += recall_score(&ids, &exact_gt[i], k);
            }
            let elapsed = start.elapsed();
            let latency = elapsed.as_secs_f64() * 1000.0 / nq as f64;
            let avg_recall = recall_sum / nq as f64;
            let vs_flat = avg_recall / flat_recall * 100.0;
            let speedup = flat_latency / latency;

            println!(
                "  {:<22} {:<8.3} {:<10.3} {:<10.1}% {:<10.2}x",
                label, avg_recall, latency, vs_flat, speedup
            );
        }
        println!();
    }
}
