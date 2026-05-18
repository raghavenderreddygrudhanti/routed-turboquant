//! Python bindings via pyo3.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use crate::index::{RoutedTQConfig, RoutedTurboQuantIndex};

/// Python-facing routed TurboQuant index.
#[pyclass(name = "RoutedTurboQuantIndex")]
pub struct PyRoutedTurboQuantIndex {
    inner: Option<RoutedTurboQuantIndex>,
    dim: usize,
    n_partitions: usize,
    n_probe: usize,
    bit_width: usize,
    kmeans_iter: usize,
    seed: u64,
}

#[pymethods]
impl PyRoutedTurboQuantIndex {
    /// Create a new (empty) index configuration.
    #[new]
    #[pyo3(signature = (dim, n_partitions=128, n_probe=8, bit_width=4, kmeans_iter=10, seed=42))]
    fn new(
        dim: usize,
        n_partitions: usize,
        n_probe: usize,
        bit_width: usize,
        kmeans_iter: usize,
        seed: u64,
    ) -> Self {
        Self {
            inner: None,
            dim,
            n_partitions,
            n_probe,
            bit_width,
            kmeans_iter,
            seed,
        }
    }

    /// Build the index from vectors array of shape (n, dim).
    fn build(&mut self, py: Python<'_>, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = vectors.shape();
        let _n = shape[0];
        let dim = shape[1];

        if dim != self.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("expected dim={}, got {}", self.dim, dim)
            ));
        }

        let data = vectors.as_slice()?;

        let n_partitions = self.n_partitions;
        let n_probe = self.n_probe;
        let bit_width = self.bit_width;
        let kmeans_iter = self.kmeans_iter;
        let seed = self.seed;

        let index = py.allow_threads(|| {
            let config = RoutedTQConfig {
                dim,
                n_partitions,
                n_probe,
                bit_width,
                kmeans_iter,
                seed,
            };
            RoutedTurboQuantIndex::build(data, config)
        });

        self.inner = Some(index);
        Ok(())
    }

    /// Search for k nearest neighbors of a single query vector.
    /// Returns (scores, indices) as numpy arrays.
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>)> {
        let index = self.inner.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("index not built"))?;

        let q = query.as_slice()?;
        let (scores, indices) = index.search(q, k);

        let scores_arr = PyArray1::from_vec(py, scores);
        let indices_arr = PyArray1::from_vec(py, indices.into_iter().map(|i| i as i64).collect());

        Ok((scores_arr, indices_arr))
    }

    /// Batch search: queries of shape (nq, dim).
    /// Returns (scores, indices) as 2D numpy arrays.
    fn search_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
        let index = self.inner.as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("index not built"))?;

        let shape = queries.shape();
        let nq = shape[0];
        let data = queries.as_slice()?;

        let (scores_vecs, indices_vecs) = py.allow_threads(|| {
            index.search_batch(data, k)
        });

        // Flatten into 2D arrays, padding with zeros if needed
        let mut scores_flat: Vec<Vec<f32>> = Vec::with_capacity(nq);
        let mut indices_flat: Vec<Vec<i64>> = Vec::with_capacity(nq);

        for i in 0..nq {
            let mut s_row = vec![0.0f32; k];
            let mut i_row = vec![-1i64; k];
            let n_results = scores_vecs[i].len().min(k);
            for j in 0..n_results {
                s_row[j] = scores_vecs[i][j];
                i_row[j] = indices_vecs[i][j] as i64;
            }
            scores_flat.push(s_row);
            indices_flat.push(i_row);
        }

        let scores_arr = PyArray2::from_vec2(py, &scores_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        let indices_arr = PyArray2::from_vec2(py, &indices_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        Ok((scores_arr, indices_arr))
    }

    /// Number of indexed vectors.
    #[getter]
    fn n_vectors(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.len())
    }

    /// Approximate memory usage in bytes.
    #[getter]
    fn memory_bytes(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.memory_bytes())
    }

    /// Scan percentage per query.
    #[getter]
    fn scan_percentage(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |i| i.scan_percentage())
    }

    /// Vectors scanned per query.
    #[getter]
    fn vectors_scanned_per_query(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.vectors_scanned_per_query())
    }
}

/// Python module definition.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRoutedTurboQuantIndex>()?;
    Ok(())
}
