# Sparse GNN Inference Accelerator Engine 
Format-Adaptive Sparse Engine (FASE): a high-performance C++ CPU accelerator for sparse matrix-matrix multiplication. Features dynamic sparse region detection, hybrid format execution (HYB), custom work-stealing scheduler, and cache-blocked SpMM for sparse GNN inference.

## Planner

| Day | Task |
|:---|:---|
| 1  | Initialize repo: folders, CMakeLists.txt, empty skeleton headers & sources. |
| 2  | Implement `types.hpp`, `utils.hpp`, `timers.hpp` (common typedefs & timer). |
| 3  | Define `csr_matrix.hpp` class skeleton (row_ptr, col_idx, vals). |
| 4  | Implement single-threaded CSR SpMM in `spmm_csr.hpp`. |
| 5  | Profile CSR: measure throughput & identify hotspots. |
| 6  | Parallelize CSR SpMM with OpenMP (`#pragma omp parallel for`). |
| 7  | Define `ell_matrix.hpp` class skeleton (fixed nnz per row). |
| 8  | Implement single-threaded ELL SpMM in `spmm_ell.hpp`. |
| 9  | Profile ELL: compare to CSR on synthetic matrices. |
| 10 | Parallelize ELL SpMM with OpenMP. |
| 11 | Define `coo_matrix.hpp` class skeleton (COO arrays). |
| 12 | Implement single-threaded COO SpMM in `spmm_coo.hpp`. |
| 13 | Profile COO: compare performance vs CSR/ELL. |
| 14 | Parallelize COO SpMM with OpenMP. |
| 15 | Research & outline region-detection algorithm. |
| 16 | Implement row-NNZ histogram in `format_detector.hpp`. |
| 17 | Classify rows into “regular” vs “irregular” bins. |
| 18 | Unit-test detector on synthetic patterns. |
| 19 | Create `hybrid_matrix.hpp` skeleton for ELL+COO hybrid storage. |
| 20 | Implement sequential hybrid SpMM in `spmm_hybrid.hpp`. |
| 21 | Profile hybrid: measure ELL vs COO portions. |
| 22 | Parallelize hybrid SpMM with OpenMP. |
| 23 | Integrate detector+hybrid into unified SpMM pipeline. |
| 24 | Test end-to-end hybrid on Cora adjacency. |
| 25 | Design cache-blocking strategy: choose tile dims. |
| 26 | Implement tiled CSR SpMM in `tiled_spmm.hpp`. |
| 27 | Profile tiled CSR vs non-tiled. |
| 28 | Implement tiled ELL SpMM in `tiled_spmm.hpp`. |
| 29 | Profile tiled ELL vs non-tiled. |
| 30 | Implement tiled hybrid SpMM in `tiled_spmm.hpp`. |
| 31 | Profile tiled hybrid vs non-tiled. |
| 32 | Benchmark all formats on Cora: CSR, ELL, COO, Hybrid, tiled. |
| 33 | Benchmark all formats on Pubmed. |
| 34 | Benchmark all formats on Reddit. |
| 35 | Integrate PAPI or perf counters to measure cache misses. |
| 36 | Use memory-access stats to adjust tile sizes. |
| 37 | Implement AVX2 intrinsics for CSR inner loop. |
| 38 | Benchmark vectorized CSR vs scalar CSR. |
| 39 | Implement AVX2 intrinsics for ELL inner loop. |
| 40 | Benchmark vectorized ELL vs scalar ELL. |
| 41 | Experiment with OpenMP chunk sizes: static vs dynamic. |
| 42 | Automate evaluation of scheduling policies. |
| 43 | Build auto-tuner to pick tile size at runtime. |
| 44 | Integrate auto-tuner into SpMM pipeline. |
| 45 | Benchmark auto-tuned vs fixed tile sizes across datasets. |
| 46 | Define and implement `bcsr_matrix.hpp` for block CSR. |
| 47 | Implement single-threaded BCSR SpMM in `spmm_bcsr.hpp`. |
| 48 | Parallelize BCSR SpMM with OpenMP. |
| 49 | Profile BCSR vs other formats. |
| 50 | Implement tiled BCSR in `tiled_spmm.hpp`. |
| 51 | Profile tiled BCSR vs non-tiled. |
| 52 | Refactor to expose unified `SpMMExecutor` API. |
| 53 | Build end-to-end runner: load graph, auto-select format, run SpMM. |
| 54 | Run runner on multiple random large graphs, collect metrics. |
| 55 | Implement custom memory pool allocator for sparse data. |
| 56 | Benchmark memory pool vs `new[]`/`malloc`. |
| 57 | Stress-test on 1M+ node synthetic graphs. |
| 58 | Gather all metrics into CSV; write results exporter. |
| 59 | Perform final performance sweep; record best configurations. |
| 60 | Tag v1.0 release, generate summary report (CSV + charts). |

---
