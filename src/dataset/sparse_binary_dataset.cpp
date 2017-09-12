#include <blitzml/dataset/sparse_binary_dataset.h>

#include <blitzml/base/math_util.h>
#include <iostream>
using std::vector;

namespace BlitzML {

SparseBinaryColumn::SparseBinaryColumn(const index_t *indices,
                                       index_t nnz, index_t length) 
    : Column(length, nnz), indices(indices) { }


value_t SparseBinaryColumn::inner_product(const vector<value_t> &vec) const {
  value_t result = 0.;
  const index_t* i;
  const index_t* end = indices + nnz();
  for (i = indices; i != end; ++i) {
    result += vec[*i];
  }
  return result;
}


value_t SparseBinaryColumn::weighted_inner_product(
    const vector<value_t> &vec, 
    const vector<value_t> &weights) const {
  value_t result = 0.;
  const index_t* i;
  const index_t* end = indices + nnz();
  for (i = indices; i != end; ++i) {
    result += vec[*i] * weights[*i];
  }
  return result;
}


value_t SparseBinaryColumn::weighted_norm_sq(
    const vector<value_t> &weights) const {
  return inner_product(weights);
}


void SparseBinaryColumn::add_multiple(vector<value_t> &target, 
                                        value_t scalar) const {
  const index_t* i;
  const index_t* end = indices + nnz();
  for (i = indices; i != end; ++i) {
    target[*i] += scalar;
  }
}


void SparseBinaryColumn::weighted_add_multiple(
    vector<value_t> &target, vector<value_t> &weights, value_t scalar) const {
  for (const index_t* i = indices; i != indices + nnz(); ++i) {
    target[*i] += scalar * weights[*i];
  }
}


value_t SparseBinaryColumn::sum() const {
  return static_cast<value_t>(nnz());
}


value_t SparseBinaryColumn::mean() const {
  return static_cast<value_t>(nnz()) / length();
}


value_t SparseBinaryColumn::l2_norm_sq() const {
  return static_cast<value_t>(nnz());
}


SparseBinaryDataset::SparseBinaryDataset(
    const index_t *indices, const size_t *indptr,
    index_t height, index_t width, size_t nnz, 
    const value_t *b, index_t length_b) 
    : Dataset(height, width, nnz, b, length_b) { 
  A_cols.clear();
  A_cols.reserve(width);
  for (index_t j = 0; j < width; ++j) {
    size_t offset = indptr[j];
    index_t col_nnz = static_cast<index_t>(indptr[j + 1] - offset);
    const index_t *col_indices = indices + offset;
    SparseBinaryColumn *col = 
              new SparseBinaryColumn(col_indices, col_nnz, height);
    A_cols.push_back(col);
  }
}


SparseBinaryDataset::~SparseBinaryDataset() {
  for (size_t ind = 0; ind < A_cols.size(); ++ind) {
    delete A_cols[ind];
  }
}


bool SparseBinaryDataset::any_columns_overlapping_in_submatrix(
    index_t first_column_submatrix, index_t end_column_submatrix) const {
  vector<bool> seen_row(height, false);
  for (index_t j = first_column_submatrix; j < end_column_submatrix; ++j) {
    const SparseBinaryColumn* col = A_cols[j];
    for (const index_t* i = col->indices_begin(); 
         i != col->indices_end(); ++i) {
      if (seen_row[*i]) {
        return true;
      } else {
        seen_row[*i] = true;
      }
    }
  }
  return false;
}

}

