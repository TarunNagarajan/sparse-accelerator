# Sparse GNN Inference Accelerator Engine 
A high-performance C++ CPU accelerator for sparse matrix-matrix multiplication. Features dynamic sparse region detection, hybrid format execution (HYB), custom work-stealing scheduler, and cache-blocked SpMM for sparse GNN inference.

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
| 60 | Tag v1.0 release, generate summary report (CSV + charts). 
| 61   | Implement AVX2 optimization for **CSR SpMM** inner loop (vectorization). |
| 62   | Benchmark **AVX2 CSR SpMM** vs scalar, profile cache misses. |
| 63   | Implement AVX2 optimization for **ELL SpMM** inner loop (vectorization). |
| 64   | Benchmark **AVX2 ELL SpMM** vs scalar, measure performance. |
| 65   | Implement AVX2 optimization for **Hybrid SpMM** kernel (combined CSR + ELL). |
| 66   | Benchmark **AVX2 Hybrid SpMM** vs scalar, analyze hotspots. |
| 67   | Implement **tiled CSR** SpMM with AVX2 support (multi-level blocking). |
| 68   | Profile **tiled CSR** SpMM with AVX2 on large random graphs. |
| 69   | Implement **tiled ELL** SpMM with AVX2 support (multi-level blocking). |
| 70   | Profile **tiled ELL** SpMM with AVX2 on random/real-world graphs. |
| 71   | Implement **tiled Hybrid** SpMM (AVX2 + multi-level blocking). |
| 72   | Final performance sweep on **CSR, ELL, Hybrid, tiled SpMM** with AVX2, measure speedups. |
| 73   | Integrate **custom memory pool allocator** for sparse matrix storage (using `malloc`-like control). |
| 74   | Profile memory pool vs `new[]`/`malloc`, measure fragmentation and access speed. |
| 75   | **Benchmarking**: Compare all sparse formats (CSR, ELL, Hybrid, tiled) on **Cora, Pubmed, Reddit**. |
| 76   | Implement **auto-tuning** for parameters (tile size, chunk sizes, thread count) via runtime profiling. |
| 77   | Integrate **auto-tuner** into SpMM pipeline and dynamically select optimal configuration per graph. |
| 78   | **Stress-test** the auto-tuner on large, real-world graphs with varying sparsity patterns. |
| 79   | **Benchmark** auto-tuned SpMM across Cora, Pubmed, Reddit, measure gains in runtime. |
| 80   | Implement **dynamic parallelism** with **OpenMP tasking** for hybrid and tiled SpMM kernels. |
| 81   | Benchmark **dynamic parallelism** implementation, compare to static parallelism. |
| 82   | Implement **thread-level parallelism** for inner-loop parallelization using OpenMP directives. |
| 83   | Test and benchmark **thread-level parallelism** vs existing optimizations. |
| 84   | **Final test suite**: Run unit tests across CSR, ELL, Hybrid, Tiled, and auto-tuned kernels. |
| 85   | **Tag v1.0 release** with all optimizations: AVX2, memory pooling, auto-tuning, dynamic parallelism. Document key performance benchmarks and results. |

---

### **Key Features:**
1. **AVX2 Optimization**: Boost kernel performance by vectorizing the inner loops for all sparse formats (CSR, ELL, Hybrid).
2. **Tiled SpMM with AVX2**: Implement advanced cache-blocking techniques with multi-level tiling, optimized for AVX2.
3. **Custom Memory Pool**: Develop a specialized allocator to reduce memory fragmentation and improve performance.
4. **Auto-Tuning**: Build an auto-tuner to dynamically select optimal parameters based on graph characteristics (tile size, chunk size, etc.).
5. **Parallelism Enhancements**: Integrate advanced **OpenMP tasking** and **thread-level parallelism** to speed up kernel execution.

---

### 🗂 Folder Structure
As the project grows, ensure the folder structure supports new features and optimizations:

```bash
src/
├── common/
│   ├── timers.hpp
│   ├── types.hpp
│   ├── utils.hpp
│   └── memory_pool.hpp   # Custom memory allocator
├── formats/
│   ├── csr_matrix.hpp
│   ├── ell_matrix.hpp
│   ├── coo_matrix.hpp
│   ├── hybrid_matrix.hpp
│   ├── bcsr_matrix.hpp
│   └── tiled_spmm.hpp    # Optimized tiled SpMM
├── kernels/
│   ├── spmm_csr.hpp      # Scalar and AVX2 SpMM
│   ├── spmm_ell.hpp      # Scalar and AVX2 SpMM
│   ├── spmm_hybrid.hpp   # Scalar and AVX2 SpMM
│   └── spmm_tiled.hpp    # Tiled AVX2 SpMM
├── tuning/
│   ├── auto_tuner.hpp    # Auto-tuning framework
│   └── parallelism.hpp   # Parallelism control (OpenMP tasking)
└── tests/
    ├── test_spmm.hpp     # Unit tests for SpMM kernels
    └── test_tuning.hpp   # Unit tests for auto-tuning and parallelism
```
