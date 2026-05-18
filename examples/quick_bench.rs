//! Quick benchmark: routed vs flat TurboQuant search.

extern crate blas_src;

use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use turbovec::TurboQuantIndex;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        for d in 0..dim {
            vecs[i * dim + d] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }
        let norm: f32 = vecs[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        for d in 0..dim {
            vecs[i * dim + d] /= norm;
        }
    }
    vecs
}

fn brute_force_topk(vectors: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<usize> {
    let n = vectors.len() / dim;
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            let sim: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            (sim, i)
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scores.iter().take(k).map(|(_, i)| *i).collect()
}

fn main() {
    let dim = 384;
    let k = 10;
    let nq = 100;

    for &n in &[10_000usize, 50_000, 100_000] {
        println!("\n============================================================");
        println!("  n={}  dim={}  k={}  nq={}", n, dim, k, nq);
        println!("============================================================");

        let vectors = random_vectors(n, dim, 42);
        let queries = random_vectors(nq, dim, 123);

        // Ground truth
        let gt: Vec<Vec<usize>> = (0..nq)
            .map(|i| brute_force_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
            .collect();

        // Flat TurboQuant
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(&vectors);
        flat_idx.prepare();

        // Warmup
        flat_idx.search(&queries[0..dim], k);

        let start = Instant::now();
        let flat_results = flat_idx.search(&queries, k);
        let flat_elapsed = start.elapsed();
        let flat_latency_ms = flat_elapsed.as_secs_f64() * 1000.0 / nq as f64;
        let flat_qps = nq as f64 / flat_elapsed.as_secs_f64();

        // Compute flat recall
        let mut flat_hits = 0;
        for i in 0..nq {
            let pred: std::collections::HashSet<usize> = flat_results.indices_for_query(i)
                .iter()
                .filter(|&&x| x >= 0)
                .map(|&x| x as usize)
                .collect();
            let truth: std::collections::HashSet<usize> = gt[i].iter().copied().collect();
            flat_hits += pred.intersection(&truth).count();
        }
        let flat_recall = flat_hits as f64 / (nq * k) as f64;

        println!("  turbovec flat:  recall={flat_recall:.3}  latency={flat_latency_ms:.3}ms  QPS={flat_qps:.0}");

        // Routed TurboQuant
        for &(n_part, n_probe) in &[(32, 4), (64, 8), (128, 8)] {
            if n_part > n / 5 {
                continue; // need at least 5 vectors per partition
            }

            let config = RoutedTQConfig {
                dim,
                n_partitions: n_part,
                n_probe: n_probe,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42, multi_assign: 1, boundary_threshold: None, max_assign: 4, rerank_top: 0,
            };

            let build_start = Instant::now();
            let routed_idx = RoutedTurboQuantIndex::build(&vectors, config);
            let build_time = build_start.elapsed();

            // Warmup
            routed_idx.search(&queries[0..dim], k);

            let start = Instant::now();
            for i in 0..nq {
                routed_idx.search(&queries[i * dim..(i + 1) * dim], k);
            }
            let routed_elapsed = start.elapsed();
            let routed_latency_ms = routed_elapsed.as_secs_f64() * 1000.0 / nq as f64;
            let routed_qps = nq as f64 / routed_elapsed.as_secs_f64();

            // Compute routed recall
            let mut routed_hits = 0;
            for i in 0..nq {
                let (_, indices) = routed_idx.search(&queries[i * dim..(i + 1) * dim], k);
                let pred: std::collections::HashSet<usize> = indices.into_iter().collect();
                let truth: std::collections::HashSet<usize> = gt[i].iter().copied().collect();
                routed_hits += pred.intersection(&truth).count();
            }
            let routed_recall = routed_hits as f64 / (nq * k) as f64;

            let scan_pct = (n_probe as f64 / n_part as f64) * 100.0;
            let speedup = flat_latency_ms / routed_latency_ms;

            println!("  routed P={n_part} R={n_probe}: recall={routed_recall:.3}  latency={routed_latency_ms:.3}ms  QPS={routed_qps:.0}  scan={scan_pct:.1}%  speedup={speedup:.1}x  build={:.2}s", build_time.as_secs_f64());
        }
    }
}
