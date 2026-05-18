//! V2 benchmark: single-index float routing vs V1 per-partition TQ vs flat.

extern crate blas_src;

use std::collections::HashSet;
use std::time::Instant;
use rand::rngs::StdRng;
use rand::SeedableRng;
use turbovec::TurboQuantIndex;
use routed_turboquant::index_v2::{RoutedV2Config, RoutedV2Index};

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

fn exact_topk(vectors: &[f32], query: &[f32], dim: usize, k: usize) -> Vec<usize> {
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

fn recall_score(predicted: &[usize], ground_truth: &[usize], k: usize) -> f64 {
    let pred_set: HashSet<usize> = predicted.iter().take(k).copied().collect();
    let gt_set: HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    pred_set.intersection(&gt_set).count() as f64 / k as f64
}

fn main() {
    let dim = 384;
    let k = 10;
    let nq = 100;

    println!("=== V2 (float routing + direct float score) vs turbovec flat ===\n");

    for &n in &[10_000, 50_000] {
        println!("================================================================");
        println!("  n={:>7}  dim={}  k={}  nq={}", n, dim, k, nq);
        println!("================================================================");

        let vectors = random_vectors(n, dim, 42);
        let queries = random_vectors(nq, dim, 123);

        // Ground truth
        let exact_gt: Vec<Vec<usize>> = (0..nq)
            .map(|i| exact_topk(&vectors, &queries[i * dim..(i + 1) * dim], dim, k))
            .collect();

        // Turbovec flat
        let mut flat_idx = TurboQuantIndex::new(dim, 4);
        flat_idx.add(&vectors);
        flat_idx.prepare();
        flat_idx.search(&queries[0..dim], k); // warmup

        let start = Instant::now();
        let flat_results = flat_idx.search(&queries, k);
        let flat_elapsed = start.elapsed();
        let flat_latency = flat_elapsed.as_secs_f64() * 1000.0 / nq as f64;

        let mut flat_recall_sum = 0.0;
        for i in 0..nq {
            let pred: Vec<usize> = flat_results.indices_for_query(i)
                .iter().filter(|&&x| x >= 0).map(|&x| x as usize).collect();
            flat_recall_sum += recall_score(&pred, &exact_gt[i], k);
        }
        let flat_recall = flat_recall_sum / nq as f64;
        println!("  turbovec flat:  recall={:.3}  latency={:.3}ms\n", flat_recall, flat_latency);

        // V2 configs
        let p = if n <= 10_000 { 32 } else if n <= 50_000 { 64 } else { 128 };

        println!("  {:<28} {:<8} {:<10} {:<10} {:<10} {:<10} {:<8}",
                 "Config", "Recall", "Latency", "Speedup", "Unique%", "Scored", "Build_s");
        println!("  {}", "-".repeat(84));

        let configs: Vec<(usize, usize, usize, usize, &str)> = vec![
            // (M, R_fraction_of_P, max_candidates, prefix_dims, label)
            // M=1 high R + partial pre-rank
            (1, p / 2, 2000, 32, "M=1 R=P/2 cap2k pd32"),
            (1, p / 2, 1000, 32, "M=1 R=P/2 cap1k pd32"),
            (1, p / 2, 500, 32, "M=1 R=P/2 cap500 pd32"),
            (1, p / 2, 1000, 64, "M=1 R=P/2 cap1k pd64"),
            (1, p / 2, 500, 64, "M=1 R=P/2 cap500 pd64"),
            (1, p * 3 / 4, 2000, 32, "M=1 R=3P/4 cap2k pd32"),
            (1, p * 3 / 4, 1000, 32, "M=1 R=3P/4 cap1k pd32"),
            (1, p * 3 / 4, 500, 32, "M=1 R=3P/4 cap500 pd32"),
            (1, p * 3 / 4, 1000, 64, "M=1 R=3P/4 cap1k pd64"),
            (1, p * 3 / 4, 500, 64, "M=1 R=3P/4 cap500 pd64"),
            // M=2 + partial pre-rank
            (2, p / 2, 1000, 32, "M=2 R=P/2 cap1k pd32"),
            (2, p / 2, 500, 32, "M=2 R=P/2 cap500 pd32"),
            (2, p * 3 / 4, 1000, 32, "M=2 R=3P/4 cap1k pd32"),
            (2, p * 3 / 4, 500, 32, "M=2 R=3P/4 cap500 pd32"),
        ];

        for &(m, r, max_cand, pdim, label) in &configs {
            let config = RoutedV2Config {
                dim, n_partitions: p, n_probe: r, kmeans_iter: 10, seed: 42,
                multi_assign: m, boundary_threshold: None, max_assign: 4,
                max_candidates: max_cand, prefix_dims: pdim,
            };

            let build_start = Instant::now();
            let idx = RoutedV2Index::build(&vectors, config);
            let build_s = build_start.elapsed().as_secs_f64();

            // Warmup
            idx.search(&queries[0..dim], k);

            // Timed search
            let start = Instant::now();
            let mut recall_sum = 0.0;
            let mut unique_sum = 0usize;
            let mut scored_sum = 0usize;
            for i in 0..nq {
                let q = &queries[i * dim..(i + 1) * dim];
                let ((_, ids), stats) = idx.search_stats(q, k);
                recall_sum += recall_score(&ids, &exact_gt[i], k);
                unique_sum += stats.unique_candidates;
                scored_sum += stats.candidates_scored;
            }
            let elapsed = start.elapsed();
            let latency = elapsed.as_secs_f64() * 1000.0 / nq as f64;
            let avg_recall = recall_sum / nq as f64;
            let speedup = flat_latency / latency;
            let avg_unique_pct = (unique_sum as f64 / nq as f64) / n as f64 * 100.0;
            let avg_scored = scored_sum as f64 / nq as f64;

            println!("  {:<28} {:<8.3} {:<10.3} {:<10.2}x {:<10.1} {:<10.0} {:<8.1}",
                     label, avg_recall, latency, speedup, avg_unique_pct, avg_scored, build_s);
        }
        println!();
    }
}
