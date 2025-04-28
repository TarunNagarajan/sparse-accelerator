# Sparse GNN Accelerator 
This repository implements FASE (Format-Adaptive Sparse Engine), a high-performance C++ accelerator for sparse matrix operations targeting Graph Neural Network (GNN) inference. It dynamically chooses between multiple sparse matrix formats (CSR, ELL, BCSR) based on sparsity patterns to maximize execution efficiency.

## Repository Structure

fase/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── architecture_overview.md
│   ├── dynamic_sparse_detection.md
│   ├── load_balancing_design.md
│   ├── cache_blocking_notes.md
│   └── benchmarks.md
├── src/
│   ├── common/
│   │   ├── types.hpp
│   │   ├── utils.hpp
│   │   └── timers.hpp
│   ├── format/
│   │   ├── csr_matrix.hpp
│   │   ├── ell_matrix.hpp
│   │   ├── coo_matrix.hpp
│   │   ├── bcsr_matrix.hpp
│   │   ├── hybrid_matrix.hpp  # <-- For dynamic ELL+COO hybrid
│   │   └── format_detector.hpp  # <-- Sparse region detection logic
│   ├── spmm/
│   │   ├── spmm_csr.hpp
│   │   ├── spmm_ell.hpp
│   │   ├── spmm_coo.hpp
│   │   ├── spmm_bcsr.hpp
│   │   ├── spmm_hybrid.hpp
│   │   └── tiled_spmm.hpp  # <-- Tiled/Cache-blocked SpMM engine
│   ├── scheduler/
│   │   ├── thread_pool.hpp  # <-- Custom threadpool
│   │   ├── work_stealing_queue.hpp
│   │   ├── task.hpp
│   │   └── scheduler.hpp
│   ├── io/
│   │   ├── graph_loader.hpp
│   │   ├── graph_converter.hpp
│   │   └── pytorch_exporter.py  # <-- Convert PyG graphs for testing
│   ├── benchmark/
│   │   ├── benchmark_runner.cpp
│   │   ├── benchmark_graphs/
│   │   │   ├── cora.mtx
│   │   │   ├── pubmed.mtx
│   │   │   └── reddit.mtx
│   └── main.cpp
├── tests/
│   ├── test_csr.cpp
│   ├── test_ell.cpp
│   ├── test_coo.cpp
│   ├── test_hybrid.cpp
│   ├── test_scheduler.cpp
│   └── test_spmm_correctness.cpp
└── scripts/
    ├── run_benchmarks.sh
    ├── format_switch_analysis.py
    ├── profile_spmm_memory.sh
    └── tuning_experiments.sh

