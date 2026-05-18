//! 500K benchmark: validate the latency crossover hypothesis.
//!
//! Generates 500K random vectors and measures turbovec flat vs routed.
//! This is the key missing proof for the crossover claim.

extern crate blas_src;

use std::collections::HashSet;
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
        for d in 0..dim { vecs[i * dim + d] /= norm; }
    }
    vecs
}

fn recall_score(predicted: &[usize], ground_truth: &[usize], k: usize) -> f64 {
    let pred_set: HashSet<usize> = predicted.iter().take(k).copied().collect();
    let gt_set: HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    pred_set.intersection(&gt_set).count() as f64 / k as f64
}

fn main() {
    let dim = 384;
    let k = 10;
    let nq = 50;

    println!("=== 500K Latency Crossover Benchmark ===");
    println!("dim={}, k={}, nq={}\n", dim, k, nq);

    // Test at 100K, 200K, 500K
    for &n in &[100_000, 200_000, 500_000] {
        println!("================================================================");
        println!("  n = {:>7}", n);
        println!("================================================================");

        println!("  Generating {} vectors...", n);
        let vectors = random_vectors(n, dim, 42);
        let queries = random_vectors(nq, dim, 123);

        // turbovec flat
        println!("  Building turbovec flat...");
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(&vectors);
        flat_idx.prepare();

        // warmup
        flat_idx.search(&queries[0..dim], k);

        let start = Instant::now();
        let _flat_results = flat_idx.search(&queries, k);
        let flat_latency = start.elapsed().as_secs_f64() * 1000.0 / nq as f64;
        println!("  turbovec flat: latency={:.3}ms", flat_latency);

        // Routed: M=1 (no duplicates, fastest build) with P=64, various R
        // Use M=1 to keep build time reasonable at 500K
        let p = 64;

        println!("  Building routed (P={}, M=1)...", p);
        let configs: Vec<(usize, &str)> = vec![
            (16, "R=16 (25%)"),
            (24, "R=24 (37.5%)"),
            (32, "R=32 (50%)"),
            (48, "R=48 (75%)"),
        ];

        for (r, label) in &configs {
            let config = RoutedTQConfig {
                dim, n_partitions: p, n_probe: *r, bit_width: 4,
                kmeans_iter: 10, seed: 42, multi_assign: 1,
                boundary_threshold: None, max_assign: 4, rerank_top: 25,
            };

            let build_start = Instant::now();
            let idx = RoutedTurboQuantIndex::build(&vectors, config);
            let build_s = build_start.elapsed().as_secs_f64();

            // warmup
            idx.search(&queries[0..dim], k);

            // timed search
            let start = Instant::now();
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                idx.search(q, k);
            }
            let latency = start.elapsed().as_secs_f64() * 1000.0 / nq as f64;
            let speedup = flat_latency / latency;
            let scan_pct = (*r as f64 / p as f64) * 100.0;

            println!("  routed {}: latency={:.3}ms  speedup={:.2}x  scan={:.0}%  build={:.0}s",
                     label, latency, speedup, scan_pct, build_s);
        }
        println!();
    }
}
