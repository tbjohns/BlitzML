#pragma once 

#include <blitzml/base/common.h>
#include <blitzml/dataset/dataset.h>

namespace BlitzML {

class SparseBinaryColumn : public Column {
  public:
    SparseBinaryColumn(const index_t *indices, index_t nnz, index_t length);

    virtual ~SparseBinaryColumn() { }

    value_t inner_product(const std::vector<value_t> &vec) const;
    value_t weighted_inner_product(
        const std::vector<value_t> &vec, 
        const std::vector<value_t> &weights) const;
    value_t weighted_norm_sq(const std::vector<value_t> &weights) const;

    void add_multiple(std::vector<value_t> &target, value_t scalar) const;
    void weighted_add_multiple(
        std::vector<value_t> &target, std::vector<value_t> &weights, value_t scalar) const;

    value_t sum() const;
    value_t mean() const;

    value_t l2_norm_sq() const;

    const index_t* indices_begin() const { return indices; }
    const index_t* indices_end() const { return indices + nnz(); }

  private:
    const index_t *indices;

    SparseBinaryColumn();
};


class SparseBinaryDataset : public Dataset {
  public:
    SparseBinaryDataset(const index_t *indices, const size_t *indptr, 
                        index_t height, index_t width, size_t nnz,
                        const value_t *b, index_t length_b);

    virtual ~SparseBinaryDataset();

    const Column* column(index_t j) const { return A_cols[j]; }

    bool any_columns_overlapping_in_submatrix(
        index_t first_column_submatrix, index_t end_column_submatrix) const;

  private:
    std::vector<SparseBinaryColumn*> A_cols;

    SparseBinaryDataset();
};


}


