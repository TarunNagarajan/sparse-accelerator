#ifndef SPMM_CSR_HPP
#define SPMM_CSR_HPP

#include "csr_matrix.hpp"
#include <vector>
#include <cassert>

namespace sparse {

/**
 * Single-threaded Sparse-Dense Matrix Multiplication (SpMM) in CSR format.
 * Computes C = A * B, where:
 *   - A is CSRMatrix<T> of size m x k
 *   - B is a dense row-major matrix of size k x n
 *   - C is the output dense row-major matrix of size m x n
 *
 * @tparam T Numeric type (e.g., float, double)
 * @param A       Input CSR matrix (m x k)
 * @param B       Input dense matrix data (flattened row-major, length k*n)
 * @param B_cols  Number of columns in B (n)
 * @param C       Output dense matrix data (flattened row-major, length m*n)
 */
template<typename T>
void spmm_csr(const CSRMatrix<T>& A,
              const std::vector<T>& B,
              std::size_t B_cols,
              std::vector<T>& C) {
    const std::size_t m = A.rows();
    const std::size_t k = A.cols();
    const std::size_t n = B_cols;
    assert(B.size() == k * n && "B size must equal k*n");

    // Allocate and zero-initialize output
    C.assign(m * n, T(0));

    const auto& row_ptr = A.row_ptr();  // length m+1
    const auto& col_idx = A.col_idx();  // length nnz
    const auto& vals    = A.vals();     // length nnz

    for (std::size_t i = 0; i < m; ++i) {
        T* C_row = C.data() + i * n;
        for (std::size_t idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const std::size_t col = col_idx[idx];
            const T value = vals[idx];
            const T* B_row = B.data() + col * n;
            for (std::size_t j = 0; j < n; ++j) {
                C_row[j] += value * B_row[j];
            }
        }
    }
}

} // namespace sparse

#endif // SPMM_CSR_HPP
