#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <iostream>
#include <cassert>

template <typename T>
class CSRMatrix {
public:
    // Constructors
    CSRMatrix() = default;

    CSRMatrix(int rows, int cols, 
              std::vector<int> row_ptr, 
              std::vector<int> col_idx, 
              std::vector<T> vals)
        : rows_(rows), cols_(cols), row_ptr_(std::move(row_ptr)), 
          col_idx_(std::move(col_idx)), vals_(std::move(vals)) 
    {
        assert(row_ptr_.size() == rows_ + 1);
        assert(col_idx_.size() == vals_.size());
    }

    // Factory from dense (slow, only for testing)
    static CSRMatrix<T> from_dense(const std::vector<std::vector<T>>& dense) {
        int rows = dense.size();
        int cols = rows > 0 ? dense[0].size() : 0;

        std::vector<int> row_ptr = {0};
        std::vector<int> col_idx;
        std::vector<T> vals;

        for (const auto& row : dense) {
            for (int j = 0; j < cols; ++j) {
                if (row[j] != T(0)) {
                    vals.push_back(row[j]);
                    col_idx.push_back(j);
                }
            }
            row_ptr.push_back(vals.size());
        }

        return CSRMatrix(rows, cols, std::move(row_ptr), std::move(col_idx), std::move(vals));
    }

    // Getters
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz() const { return vals_.size(); }

    const std::vector<int>& row_ptr() const { return row_ptr_; }
    const std::vector<int>& col_idx() const { return col_idx_; }
    const std::vector<T>& vals() const { return vals_; }

    // Print (debug)
    void print() const {
        std::cout << "CSR Matrix (" << rows_ << " x " << cols_ << "), nnz=" << nnz() << "\n";
        std::cout << "row_ptr: ";
        for (int i : row_ptr_) std::cout << i << " ";
        std::cout << "\ncol_idx: ";
        for (int i : col_idx_) std::cout << i << " ";
        std::cout << "\nvals: ";
        for (T v : vals_) std::cout << v << " ";
        std::cout << "\n";
    }

    // Access single row as (col, val) pairs
    std::vector<std::pair<int, T>> get_row(int i) const {
        assert(i >= 0 && i < rows_);
        std::vector<std::pair<int, T>> result;
        for (int j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
            result.emplace_back(col_idx_[j], vals_[j]);
        }
        return result;
    }

private:
    int rows_ = 0;
    int cols_ = 0;

    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;
    std::vector<T> vals_;
};

#endif // CSR_MATRIX_HPP
