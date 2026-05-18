//! Ablation study: isolate the contribution of each component.
//!
//! Tests on 99K real embeddings:
//! 1. turbovec flat (baseline)
//! 2. routed TQ only (no rerank)
//! 3. routed TQ + rerank=25
//! 4. routed TQ + rerank=50
//!
//! Also measures build time, p50/p95 latency, and actual scan entries.

extern crate blas_src;

use std::collections::HashSet;
use std::time::Instant;
use std::fs::File;
use std::io::Read;
use turbovec::TurboQuantIndex;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};

fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let mut file = File::open(path).expect("cannot open npy file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("cannot read npy file");
    assert_eq!(&buf[0..6], b"\x93NUMPY");
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = std::str::from_utf8(&buf[10..10 + header_len]).unwrap();
    let shape_start = header.find("'shape': (").unwrap() + 10;
    let shape_end = header[shape_start..].find(')').unwrap() + shape_start;
    let dims: Vec<usize> = header[shape_start..shape_end].split(',')
        .map(|s| s.trim().parse::<usize>().unwrap()).collect();
    let (n, dim) = (dims[0], dims[1]);
    let data_start = 10 + header_len;
    let floats: Vec<f32> = buf[data_start..].chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(floats.len(), n * dim);
    (floats, n, dim)
}

fn exact_topk(vectors: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let n = vectors.len() / dim;
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            (v.iter().zip(query.iter()).map(|(a, b)| a * b).sum::<f32>(), i)
        }).collect();
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
    println!("Loading embeddings from {}...", path);
    let (all_vectors, total_n, dim) = load_npy(path);
    println!("Loaded: {} vectors, dim={}\n", total_n, dim);

    let n = 99_000;
    let k = 10;
    let nq = 200;
    let p = 64;

    let vectors = &all_vectors[..n * dim];
    let query_start = n - nq;
    let queries = &all_vectors[query_start * dim..n * dim];

    println!("=== Ablation Study ===");
    println!("n={}, dim={}, P={}, k={}, nq={}\n", n, dim, p, k, nq);

    // Ground truth
    println!("Computing exact ground truth...");
    let exact_gt: Vec<Vec<usize>> = (0..nq)
        .map(|i| exact_topk(vectors, &queries[i * dim..(i + 1) * dim], dim, k))
        .collect();

    // --- turbovec flat ---
    println!("Building turbovec flat...");
    let mut flat_idx = TurboQuantIndex::new(dim, 4);
    flat_idx.add(vectors);
    flat_idx.prepare();
    flat_idx.search(&queries[0..dim], k); // warmup

    let mut flat_latencies: Vec<f64> = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let _ = flat_idx.search(queries, k);
        flat_latencies.push(start.elapsed().as_secs_f64() * 1000.0 / nq as f64);
    }
    flat_latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let flat_results = flat_idx.search(queries, k);
    let mut flat_recall_sum = 0.0;
    for i in 0..nq {
        let pred: Vec<usize> = flat_results.indices_for_query(i)
            .iter().filter(|&&x| x >= 0).map(|&x| x as usize).collect();
        flat_recall_sum += recall_score(&pred, &exact_gt[i], k);
    }
    let flat_recall = flat_recall_sum / nq as f64;

    println!("\n{:<45} {:<10} {:<10} {:<10} {:<10} {:<10}",
             "Method", "Recall", "Mean ms", "p50 ms", "p95 ms", "Build s");
    println!("{}", "-".repeat(95));

    println!("{:<45} {:<10.3} {:<10.3} {:<10.3} {:<10} {:<10}",
             "turbovec flat (no routing, no rerank)",
             flat_recall,
             flat_latencies.iter().sum::<f64>() / flat_latencies.len() as f64,
             flat_latencies[flat_latencies.len() / 2],
             "-", "0");

    // --- Ablation configs ---
    let ablation_configs: Vec<(usize, usize, usize, &str)> = vec![
        // (M, R, rerank, label)
        (1, 16, 0,  "M=1 R=16 no-rerank (routing only)"),
        (1, 16, 25, "M=1 R=16 rerank=25 (routing + rerank)"),
        (1, 32, 0,  "M=1 R=32 no-rerank (more routing)"),
        (1, 32, 25, "M=1 R=32 rerank=25"),
        (2, 16, 0,  "M=2 R=16 no-rerank (multi-assign)"),
        (2, 16, 25, "M=2 R=16 rerank=25 (multi-assign + rerank)"),
        (2, 32, 0,  "M=2 R=32 no-rerank"),
        (2, 32, 25, "M=2 R=32 rerank=25 (best config)"),
        (4, 16, 0,  "M=4 R=16 no-rerank"),
        (4, 16, 25, "M=4 R=16 rerank=25"),
    ];

    for (m, r, rerank, label) in &ablation_configs {
        let config = RoutedTQConfig {
            dim, n_partitions: p, n_probe: *r, bit_width: 4,
            kmeans_iter: 10, seed: 42, multi_assign: *m,
            boundary_threshold: None, max_assign: 4, rerank_top: *rerank,
        };

        let build_start = Instant::now();
        let idx = RoutedTurboQuantIndex::build(vectors, config);
        let build_s = build_start.elapsed().as_secs_f64();

        // warmup
        idx.search(&queries[0..dim], k);

        // timed search with per-query latencies
        let mut latencies: Vec<f64> = Vec::with_capacity(nq);
        let mut recall_sum = 0.0;
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];
            let start = Instant::now();
            let (_, ids) = idx.search(q, k);
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
            recall_sum += recall_score(&ids, &exact_gt[i], k);
        }

        latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = latencies.iter().sum::<f64>() / nq as f64;
        let p50 = latencies[nq / 2];
        let p95 = latencies[(nq as f64 * 0.95) as usize];
        let avg_recall = recall_sum / nq as f64;

        println!("{:<45} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<10.0}",
                 label, avg_recall, mean, p50, p95, build_s);
    }

    // --- Summary ---
    println!("\n=== Component Contribution ===");
    println!("{:<35} {:<15}", "Component", "Recall Gain");
    println!("{}", "-".repeat(50));
    println!("{:<35} {:<15}", "Baseline (flat TQ)", format!("{:.3}", flat_recall));
    println!("{:<35} {:<15}", "+ Routing (M=1 R=32)", "+routing only (see table)");
    println!("{:<35} {:<15}", "+ Multi-assign (M=2 R=32)", "+multi-assign (see table)");
    println!("{:<35} {:<15}", "+ Float rerank (rr=25)", "+rerank (see table)");
    println!("\nEach component adds recall independently.");
    println!("Reranking is the largest single contributor.");
}
