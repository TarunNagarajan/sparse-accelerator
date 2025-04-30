#pragma once

#include <cstdint> 
// fixed-width integer types, like uint32_t

#include <cstddef>
// size_t

/**
 * @brief index_t
 * 
 * A 32-bit unsigned integer type used throughout to index
 * rows, columns, and non-zero entries in sparse matrices. 
 * 
 * Chosen to be large enough for graphs with up to ~4 billion nodes/edges
 * while keeping memory footprint low.
 */

using index_t = uint32_t; 

/**
 * @brief value_t
 * 
 * A floating-point type for storing non-zero matrix values
 * and performing arithmetic. We use `float` to balance precision
 * (sufficient for many ML workloads) with memory bandwidth and
 * cache efficiency.
 */

using value_t = float;

constexpr index_t INVALID_INDEX = static_cast<index_t>(-1); 

// - Every sparse format class (CSR, ELL, Hybrid, BCSR) uses 
//   index_t for its row/column pointers and indices, ensuring
//   uniformity across modules.

// - All kernel implementations (SpMM, tiled SpMM, vectorized loops)
//   use value_t for arithmetic, so changing precision everywhere
//   is as simple as swapping this typedef.

// - INVALID_INDEX helps signal errors or empty slots in various
//   routines (e.g., when splitting a matrix into ELL vs COO regions).
