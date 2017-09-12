#include <blitzml/dataset/sparse_dataset.h>

#include <blitzml/base/math_util.h>

using std::vector;

namespace BlitzML {

template <typename data_t>
SparseColumn<data_t>::SparseColumn(const index_t *indices, const data_t *values, 
                                   index_t nnz, index_t length) 
    : Column(length, nnz), indices(indices), values(values) { }


template <typename data_t>
value_t SparseColumn<data_t>::inner_product(const vector<value_t> &vec) const {
  value_t result = 0.;
  for (index_t ind = 0; ind < nnz_; ++ind) {
    result += values[ind] * vec[indices[ind]];
  }
  return result;
}


template <typename data_t>
value_t SparseColumn<data_t>::weighted_inner_product(
    const vector<value_t> &vec, 
    const vector<value_t> &weights) const {
  value_t result = 0.;
  for (index_t ind = 0; ind < nnz_; ++ind) {
    index_t i = indices[ind];
    result += values[ind] * vec[i] * weights[i];
  }
  return result;
}


template <typename data_t>
value_t SparseColumn<data_t>::weighted_norm_sq(
    const vector<value_t> &weights) const {
  value_t result = 0.;
  for (index_t ind = 0; ind < nnz_; ++ind) {
    result += sq(values[ind]) * weights[indices[ind]];
  }
  return result;
}


template <typename data_t>
void SparseColumn<data_t>::add_multiple(vector<value_t> &target, 
                                        value_t scalar) const {
                                            
  const index_t* i = indices;
  const data_t* v = values;
  for (index_t n = nnz_; n != 0; --n) {
    target[*i++] += (*v++) * scalar;
  }
}


template <typename data_t>
void SparseColumn<data_t>::weighted_add_multiple(
    vector<value_t> &target, vector<value_t> &weights, value_t scalar) const {
  const index_t* i = indices;
  const data_t* v = values;
  for (index_t n = nnz_; n != 0; --n) {
    target[*i] += (*v) * scalar * weights[*i];
    ++i; ++v;
  }
}


template <typename data_t>
value_t SparseColumn<data_t>::sum() const {
  return sum_array(values, nnz_);
}


template <typename data_t>
value_t SparseColumn<data_t>::mean() const {
  return sum() / length();
}


template <typename data_t>
value_t SparseColumn<data_t>::l2_norm_sq() const {
  return BlitzML::l2_norm_sq(values, nnz_);
}


template class SparseColumn<float>;
template class SparseColumn<double>;
template class SparseColumn<int>;
template class SparseColumn<bool>;



template <typename data_t>
SparseDataset<data_t>::SparseDataset(
    const index_t *indices, const size_t *indptr, const data_t *data, 
    index_t height, index_t width, size_t nnz, 
    const value_t *b, index_t length_b) 
    : Dataset(height, width, nnz, b, length_b) { 

  A_cols.resize(width);
  for (index_t j = 0; j < width; ++j) {
    size_t offset = indptr[j];
    index_t col_nnz = static_cast<index_t>(indptr[j + 1] - offset);
    const data_t* col_data = data + offset;
    const index_t* col_indices = indices + offset;
    A_cols[j] = SparseColumn<data_t>(col_indices, col_data, col_nnz, height);
  }
}


template <typename data_t>
bool SparseDataset<data_t>::any_columns_overlapping_in_submatrix(
    index_t first_column_submatrix, index_t end_column_submatrix) const {
  vector<bool> seen_row(height, false);
  for (index_t j = first_column_submatrix; j < end_column_submatrix; ++j) {
    const SparseColumn<data_t>& col = A_cols[j];
    for (const index_t* i = col.indices_begin(); 
         i != col.indices_end(); ++i) {
      if (seen_row[*i]) {
        return true;
      } else {
        seen_row[*i] = true;
      }
    }
  }
  return false;
}


template class SparseDataset<float>;
template class SparseDataset<double>;
template class SparseDataset<int>;
template class SparseDataset<bool>;


}
