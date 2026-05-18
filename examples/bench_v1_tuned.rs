//! V1 tuned: find best M/R combination that maximizes recall while minimizing
//! duplicate overhead. Compare M=1,2,3,4 at various R with rerank=25.

extern crate blas_src;

use rand::rngs::StdRng;
use rand::SeedableRng;
use routed_turboquant::index::{RoutedTQConfig, RoutedTurboQuantIndex};
use std::collections::HashSet;
use std::time::Instant;
use turbovec::TurboQuantIndex;

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vecs = vec![0.0f32; n * dim];
    for i in 0..n {
        for d in 0..dim {
            vecs[i * dim + d] = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        }
        let norm: f32 = vecs[i * dim..(i + 1) * dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        for d in 0..dim {
            vecs[i * dim + d] /= norm;
        }
    }
    vecs
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
    let dim = 384;
    let k = 10;
    let nq = 100;
    let n = 10_000;
    let p = 32;

    println!(
        "V1 Tuned: n={}, dim={}, P={}, k={}, nq={}",
        n, dim, p, k, nq
    );

    let vectors = random_vectors(n, dim, 42);
    let queries = random_vectors(nq, dim, 123);

    let exact_gt: Vec<Vec<usize>> = (0..nq)
        .map(|i| exact_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
        .collect();

    // Flat baseline
    let mut flat_idx = TurboQuantIndex::new(dim, 4);
    flat_idx.add(&vectors);
    flat_idx.prepare();
    flat_idx.search(&queries[0..dim], k);
    let start = Instant::now();
    let flat_results = flat_idx.search(&queries, k);
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
    println!(
        "turbovec flat: recall={:.3} latency={:.3}ms\n",
        flat_recall_sum / nq as f64,
        flat_latency
    );

    println!(
        "{:<6} {:<6} {:<8} {:<10} {:<10} {:<10} {:<10}",
        "M", "R", "Scan%", "Recall", "Latency", "Speedup", "StorageX"
    );
    println!("{}", "-".repeat(60));

    for &m in &[1, 2, 3, 4] {
        for &r in &[8, 12, 16, 20, 24, 32] {
            let config = RoutedTQConfig {
                dim,
                n_partitions: p,
                n_probe: r,
                bit_width: 4,
                kmeans_iter: 10,
                seed: 42,
                multi_assign: m,
                boundary_threshold: None,
                max_assign: 4,
                rerank_top: 25,
            };

            let idx = RoutedTurboQuantIndex::build(&vectors, config);

            // Warmup
            idx.search(&queries[0..dim], k);

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
            let speedup = flat_latency / latency;
            let scan_pct = (r as f64 / p as f64) * 100.0;

            println!(
                "{:<6} {:<6} {:<8.1} {:<10.3} {:<10.3} {:<10.2}x {:<10}",
                m,
                r,
                scan_pct,
                avg_recall,
                latency,
                speedup,
                format!("{}x", m)
            );
        }
        println!();
    }
}
