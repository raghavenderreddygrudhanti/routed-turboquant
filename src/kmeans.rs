//! Float-space k-means with k-means++ initialization.
//!
//! Operates on unit-normalized vectors using inner product as similarity.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};

/// K-means result: centroids and per-vector partition assignments.
pub struct KMeansResult {
    /// Centroids of shape (k, dim), unit-normalized.
    pub centroids: Vec<Vec<f32>>,
    /// Partition assignment for each input vector.
    pub assignments: Vec<u32>,
}

/// Run k-means++ on unit-normalized vectors.
///
/// Uses inner product as similarity (equivalent to cosine on unit vectors).
/// Returns centroids and assignments.
pub fn kmeans_float(
    vectors: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> KMeansResult {
    assert!(n > 0 && dim > 0 && k > 0);
    assert!(vectors.len() == n * dim);

    let mut rng = StdRng::seed_from_u64(seed);

    // k-means++ initialization
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // First centroid: random vector
    let first_idx = rand::Rng::gen_range(&mut rng, 0..n);
    centroids.push(vectors[first_idx * dim..(first_idx + 1) * dim].to_vec());

    for _ in 1..k {
        // Compute max similarity to existing centroids for each vector
        let mut probs: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let max_sim = centroids.iter()
                .map(|c| dot(v, c))
                .fold(f32::NEG_INFINITY, f32::max);
            // Probability proportional to (1 - max_sim), clamped >= 0
            let p = (1.0 - max_sim).max(0.0) as f64;
            probs.push(p);
        }

        let sum: f64 = probs.iter().sum();
        if sum < 1e-12 {
            // All vectors are identical to existing centroids; pick random
            let idx = rand::Rng::gen_range(&mut rng, 0..n);
            centroids.push(vectors[idx * dim..(idx + 1) * dim].to_vec());
        } else {
            let dist = WeightedIndex::new(&probs).unwrap();
            let idx = dist.sample(&mut rng);
            centroids.push(vectors[idx * dim..(idx + 1) * dim].to_vec());
        }
    }

    // Iterative assignment + update
    let mut assignments = vec![0u32; n];

    for _iter in 0..max_iter {
        // Assign each vector to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let mut best_c = 0u32;
            let mut best_sim = f32::NEG_INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let sim = dot(v, centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_c = c as u32;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];

        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            let v = &vectors[i * dim..(i + 1) * dim];
            for d in 0..dim {
                sums[c][d] += v[d] as f64;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                let mut centroid: Vec<f32> = sums[c].iter().map(|x| (x * inv) as f32).collect();
                // Normalize centroid
                let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    for x in centroid.iter_mut() {
                        *x /= norm;
                    }
                }
                centroids[c] = centroid;
            } else {
                // Empty cluster: reinitialize to random vector
                let idx = rand::Rng::gen_range(&mut rng, 0..n);
                centroids[c] = vectors[idx * dim..(idx + 1) * dim].to_vec();
            }
        }
    }

    KMeansResult {
        centroids,
        assignments,
    }
}

/// Dot product of two slices.
#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let dim = 4;
        let n = 100;
        let k = 4;

        // Generate clustered data
        let mut vectors = vec![0.0f32; n * dim];
        let mut rng = StdRng::seed_from_u64(123);
        for i in 0..n {
            let cluster = i % k;
            for d in 0..dim {
                vectors[i * dim + d] = if d == cluster {
                    1.0
                } else {
                    rand::Rng::gen_range(&mut rng, -0.1..0.1)
                };
            }
            // Normalize
            let norm: f32 = vectors[i * dim..(i + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            for d in 0..dim {
                vectors[i * dim + d] /= norm;
            }
        }

        let result = kmeans_float(&vectors, n, dim, k, 20, 42);
        assert_eq!(result.assignments.len(), n);
        assert_eq!(result.centroids.len(), k);

        // Check all assignments are valid
        for &a in &result.assignments {
            assert!((a as usize) < k);
        }
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let dim = 8;
        let n = 50;
        let k = 1;

        let mut vectors = vec![0.0f32; n * dim];
        for i in 0..n {
            vectors[i * dim] = 1.0; // all point in same direction
        }

        let result = kmeans_float(&vectors, n, dim, k, 10, 42);
        assert!(result.assignments.iter().all(|&a| a == 0));
    }
}
