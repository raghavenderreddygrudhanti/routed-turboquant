//! V2 Architecture: float routing + partial dot pre-rank + full float rerank.
//!
//! Pipeline:
//!   1. Float k-means partitions (routing)
//!   2. Partition membership lists (no vector duplication)
//!   3. At query time:
//!      a. Route to top-R partitions (P centroid dot products)
//!      b. Collect unique candidate IDs from those partitions
//!      c. Pre-rank candidates with partial dot product (first prefix_dims dims)
//!      d. Full float score top max_candidates
//!      e. Return top-k

use rayon::prelude::*;
use crate::kmeans::{kmeans_float, KMeansResult};

/// V2 config.
pub struct RoutedV2Config {
    pub dim: usize,
    pub n_partitions: usize,
    pub n_probe: usize,
    pub kmeans_iter: usize,
    pub seed: u64,
    pub multi_assign: usize,
    pub boundary_threshold: Option<f32>,
    pub max_assign: usize,
    /// Budget: how many candidates to full-float-score after pre-ranking.
    /// 0 = score all (no cap).
    pub max_candidates: usize,
    /// Dimensions for partial dot product pre-ranking (e.g. 32, 48, 64).
    /// 0 = no partial pre-rank.
    pub prefix_dims: usize,
}

impl Default for RoutedV2Config {
    fn default() -> Self {
        Self {
            dim: 384,
            n_partitions: 128,
            n_probe: 8,
            kmeans_iter: 10,
            seed: 42,
            multi_assign: 1,
            boundary_threshold: None,
            max_assign: 4,
            max_candidates: 500,
            prefix_dims: 32,
        }
    }
}

/// V2 routed index.
pub struct RoutedV2Index {
    config: RoutedV2Config,
    centroids: Vec<f32>,
    vectors: Vec<f32>,
    partitions: Vec<Vec<u32>>,
    n_vectors: usize,
    total_assignments: usize,
}

/// V2 search stats.
#[derive(Debug, Clone, Default)]
pub struct V2SearchStats {
    pub unique_candidates: usize,
    pub candidates_scored: usize,
}

impl RoutedV2Index {
    pub fn build(vectors: &[f32], config: RoutedV2Config) -> Self {
        let n = vectors.len() / config.dim;
        assert_eq!(vectors.len(), n * config.dim);
        assert!(n > 0);
        let dim = config.dim;

        // Normalize for clustering
        let mut unit_vectors = vectors.to_vec();
        for i in 0..n {
            let s = i * dim;
            let slice = &mut unit_vectors[s..s + dim];
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 { for x in slice.iter_mut() { *x /= norm; } }
        }

        let KMeansResult { centroids, .. } = kmeans_float(
            &unit_vectors, n, dim, config.n_partitions, config.kmeans_iter, config.seed,
        );
        let flat_centroids: Vec<f32> = centroids.iter().flatten().copied().collect();

        // Assign vectors to partitions
        let mut partitions: Vec<Vec<u32>> = vec![Vec::new(); config.n_partitions];
        let mut total_assignments = 0usize;

        if let Some(threshold) = config.boundary_threshold {
            let max_m = config.max_assign.min(config.n_partitions);
            for i in 0..n {
                let v = &unit_vectors[i * dim..(i + 1) * dim];
                let mut scores: Vec<(f32, usize)> = (0..config.n_partitions)
                    .map(|p| {
                        let c = &flat_centroids[p * dim..(p + 1) * dim];
                        (v.iter().zip(c.iter()).map(|(a, b)| a * b).sum::<f32>(), p)
                    }).collect();
                scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let best = scores[0].0;
                let cutoff = best - threshold;
                partitions[scores[0].1].push(i as u32);
                total_assignments += 1;
                for &(sim, pid) in scores.iter().skip(1).take(max_m - 1) {
                    if sim >= cutoff { partitions[pid].push(i as u32); total_assignments += 1; }
                    else { break; }
                }
            }
        } else {
            let m = config.multi_assign;
            for i in 0..n {
                let v = &unit_vectors[i * dim..(i + 1) * dim];
                let mut scores: Vec<(f32, usize)> = (0..config.n_partitions)
                    .map(|p| {
                        let c = &flat_centroids[p * dim..(p + 1) * dim];
                        (v.iter().zip(c.iter()).map(|(a, b)| a * b).sum::<f32>(), p)
                    }).collect();
                scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                for &(_, pid) in scores.iter().take(m) {
                    partitions[pid].push(i as u32);
                    total_assignments += 1;
                }
            }
        }

        Self { config, centroids: flat_centroids, vectors: vectors.to_vec(), partitions, n_vectors: n, total_assignments }
    }

    /// Search: route → collect → partial pre-rank → full score → top-k.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
        assert_eq!(query.len(), self.config.dim);
        if self.n_vectors == 0 { return (vec![], vec![]); }

        let dim = self.config.dim;
        let n_probe = self.config.n_probe.min(self.config.n_partitions);

        // Route
        let mut query_unit = query.to_vec();
        let norm: f32 = query_unit.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 { for x in query_unit.iter_mut() { *x /= norm; } }

        let mut cs: Vec<(f32, usize)> = (0..self.config.n_partitions)
            .map(|p| {
                let c = &self.centroids[p * dim..(p + 1) * dim];
                (query_unit.iter().zip(c.iter()).map(|(a, b)| a * b).sum::<f32>(), p)
            }).collect();
        cs.select_nth_unstable_by(n_probe - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());

        // Collect unique IDs
        let mut visited = vec![false; self.n_vectors];
        let mut cands: Vec<u32> = Vec::new();
        for &(_, pid) in cs[..n_probe].iter() {
            for &vid in &self.partitions[pid] {
                if !visited[vid as usize] { visited[vid as usize] = true; cands.push(vid); }
            }
        }
        if cands.is_empty() { return (vec![], vec![]); }

        // Pre-rank with partial dot product, then full-score top N
        let max_cand = if self.config.max_candidates > 0 { self.config.max_candidates } else { cands.len() };

        let to_score = if cands.len() > max_cand && self.config.prefix_dims > 0 {
            let pdim = self.config.prefix_dims.min(dim);
            let qp = &query[..pdim];
            let mut ps: Vec<(f32, u32)> = cands.iter().map(|&vid| {
                let v = &self.vectors[vid as usize * dim..vid as usize * dim + pdim];
                (qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>(), vid)
            }).collect();
            ps.select_nth_unstable_by(max_cand - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
            ps[..max_cand].iter().map(|&(_, v)| v).collect::<Vec<_>>()
        } else {
            cands
        };

        // Full float score
        let mut scored: Vec<(f32, usize)> = to_score.iter().map(|&vid| {
            let v = &self.vectors[vid as usize * dim..(vid as usize + 1) * dim];
            (query.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>(), vid as usize)
        }).collect();

        if scored.len() > k {
            scored.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
            scored.truncate(k);
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        } else {
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }

        (scored.iter().map(|(s, _)| *s).collect(), scored.iter().map(|(_, i)| *i).collect())
    }

    /// Search with stats.
    pub fn search_stats(&self, query: &[f32], k: usize) -> ((Vec<f32>, Vec<usize>), V2SearchStats) {
        assert_eq!(query.len(), self.config.dim);
        if self.n_vectors == 0 { return ((vec![], vec![]), V2SearchStats::default()); }

        let dim = self.config.dim;
        let n_probe = self.config.n_probe.min(self.config.n_partitions);

        let mut query_unit = query.to_vec();
        let norm: f32 = query_unit.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 { for x in query_unit.iter_mut() { *x /= norm; } }

        let mut cs: Vec<(f32, usize)> = (0..self.config.n_partitions)
            .map(|p| {
                let c = &self.centroids[p * dim..(p + 1) * dim];
                (query_unit.iter().zip(c.iter()).map(|(a, b)| a * b).sum::<f32>(), p)
            }).collect();
        cs.select_nth_unstable_by(n_probe - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut visited = vec![false; self.n_vectors];
        let mut cands: Vec<u32> = Vec::new();
        for &(_, pid) in cs[..n_probe].iter() {
            for &vid in &self.partitions[pid] {
                if !visited[vid as usize] { visited[vid as usize] = true; cands.push(vid); }
            }
        }
        let unique_candidates = cands.len();
        if unique_candidates == 0 { return ((vec![], vec![]), V2SearchStats { unique_candidates: 0, candidates_scored: 0 }); }

        let max_cand = if self.config.max_candidates > 0 { self.config.max_candidates } else { cands.len() };

        let to_score = if cands.len() > max_cand && self.config.prefix_dims > 0 {
            let pdim = self.config.prefix_dims.min(dim);
            let qp = &query[..pdim];
            let mut ps: Vec<(f32, u32)> = cands.iter().map(|&vid| {
                let v = &self.vectors[vid as usize * dim..vid as usize * dim + pdim];
                (qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>(), vid)
            }).collect();
            ps.select_nth_unstable_by(max_cand - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
            ps[..max_cand].iter().map(|&(_, v)| v).collect::<Vec<_>>()
        } else {
            cands
        };

        let candidates_scored = to_score.len();
        let mut scored: Vec<(f32, usize)> = to_score.iter().map(|&vid| {
            let v = &self.vectors[vid as usize * dim..(vid as usize + 1) * dim];
            (query.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>(), vid as usize)
        }).collect();

        if scored.len() > k {
            scored.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap());
            scored.truncate(k);
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        } else {
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        }

        let r = (scored.iter().map(|(s, _)| *s).collect(), scored.iter().map(|(_, i)| *i).collect());
        (r, V2SearchStats { unique_candidates, candidates_scored })
    }

    pub fn search_batch(&self, queries: &[f32], k: usize) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let nq = queries.len() / self.config.dim;
        let results: Vec<_> = (0..nq).into_par_iter()
            .map(|i| self.search(&queries[i * self.config.dim..(i + 1) * self.config.dim], k))
            .collect();
        (results.iter().map(|(s, _)| s.clone()).collect(), results.iter().map(|(_, i)| i.clone()).collect())
    }

    pub fn len(&self) -> usize { self.n_vectors }
    pub fn is_empty(&self) -> bool { self.n_vectors == 0 }
    pub fn storage_factor(&self) -> f64 { self.total_assignments as f64 / self.n_vectors as f64 }
    pub fn scan_percentage(&self) -> f64 { (self.config.n_probe as f64 / self.config.n_partitions as f64) * 100.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut vecs = vec![0.0f32; n * dim];
        for i in 0..n {
            for d in 0..dim { vecs[i * dim + d] = rand::Rng::gen_range(&mut rng, -1.0..1.0); }
            let norm: f32 = vecs[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            for d in 0..dim { vecs[i * dim + d] /= norm; }
        }
        vecs
    }

    #[test]
    fn test_v2_basic() {
        let dim = 32;
        let n = 500;
        let vectors = random_vectors(n, dim, 42);
        let config = RoutedV2Config { dim, n_partitions: 8, n_probe: 8, multi_assign: 1, max_candidates: 0, prefix_dims: 0, ..Default::default() };
        let idx = RoutedV2Index::build(&vectors, config);
        let (scores, indices) = idx.search(&vectors[0..dim], 10);
        assert_eq!(indices[0], 0);
        assert!(scores.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn test_v2_partial_prerank() {
        let dim = 64;
        let n = 1000;
        let vectors = random_vectors(n, dim, 42);
        // With partial pre-rank, cap500 should still find reasonable results
        let config = RoutedV2Config { dim, n_partitions: 8, n_probe: 8, multi_assign: 1, max_candidates: 500, prefix_dims: 16, ..Default::default() };
        let idx = RoutedV2Index::build(&vectors, config);
        let (_, indices) = idx.search(&vectors[0..dim], 10);
        // Self should be in top-10 with full probe
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_v2_no_duplicates() {
        let dim = 32;
        let n = 200;
        let vectors = random_vectors(n, dim, 42);
        let config = RoutedV2Config { dim, n_partitions: 4, n_probe: 4, multi_assign: 3, max_candidates: 0, prefix_dims: 0, ..Default::default() };
        let idx = RoutedV2Index::build(&vectors, config);
        let (_, indices) = idx.search(&vectors[0..dim], 10);
        let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), indices.len());
    }
}
