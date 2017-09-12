#include <blitzml/dataset/dense_dataset.h>

#include <blitzml/base/math_util.h>

using std::vector;

namespace BlitzML {


size_t coordinates2dense_array_index(index_t row, index_t col, index_t height) {
  size_t offset = static_cast<size_t>(height) * static_cast<size_t>(col);
  return offset + static_cast<size_t>(row);
}


template <typename data_t>
DenseColumn<data_t>::DenseColumn(const data_t *values, index_t length) 
    : Column(length, length), values(values) { }


template <typename data_t>
value_t DenseColumn<data_t>::inner_product(const vector<value_t> &vec) const {
  value_t result = 0.0;
  for (index_t i = 0; i < length(); ++i)
    result += values[i] * vec[i];
  return result;
}


template <typename data_t>
value_t DenseColumn<data_t>::weighted_inner_product(
    const vector<value_t> &vec, const vector<value_t> &weights) const {
  value_t result = 0.0;
  for (index_t i = 0; i < length(); ++i)
    result += values[i] * vec[i] * weights[i];
  return result;
}


template <typename data_t>
value_t DenseColumn<data_t>::weighted_norm_sq(
    const vector<value_t> &weights) const {

  value_t result = 0.0;
  for (index_t i = 0; i < length(); ++i)
    result += values[i] * values[i] * weights[i];
  return result;
}


template <typename data_t>
void DenseColumn<data_t>::add_multiple(
    vector<value_t> &target, value_t scalar) const {
  for (index_t i = 0; i < length(); ++i) {
    target[i] += values[i] * scalar;
  }
}


template <typename data_t>
void DenseColumn<data_t>::weighted_add_multiple(
    std::vector<value_t> &target, std::vector<value_t> &weights, value_t scalar) const {
  for (index_t i = 0; i < length(); ++i) {
    target[i] += values[i] * weights[i] * scalar;
  }
}


template <typename data_t>
value_t DenseColumn<data_t>::sum() const {
  return sum_array(values, length());
}


template <typename data_t>
value_t DenseColumn<data_t>::mean() const {
  return this->sum() / length();
}


template <typename data_t>
value_t DenseColumn<data_t>::l2_norm_sq() const {
  return BlitzML::l2_norm_sq(values, length());
}

template class DenseColumn<double>;
template class DenseColumn<float>;
template class DenseColumn<int>;
template class DenseColumn<bool>;



template<typename data_t>
DenseDataset<data_t>::DenseDataset(const data_t *data, 
    index_t height, index_t width, const value_t *b, index_t length_b) 
    : Dataset(height, width, 
              coordinates2dense_array_index(0, width, height),
              b, length_b) {
  A_cols.clear();
  A_cols.reserve(width);
  for (index_t col_ind = 0; col_ind < width; ++col_ind) {
    size_t array_index = coordinates2dense_array_index(0, col_ind, height);
    const data_t *col_data = data + array_index;
    DenseColumn<data_t>* col = new DenseColumn<data_t>(col_data, height);
    A_cols.push_back(col);
  }
}


template <typename data_t>
DenseDataset<data_t>::~DenseDataset() {
  for (size_t ind = 0; ind < A_cols.size(); ++ind) {
    delete A_cols[ind];
  }
}


template<typename data_t>
bool DenseDataset<data_t>::any_columns_overlapping_in_submatrix(
    index_t first_column_submatrix, index_t end_column_submatrix) const {
  if (end_column_submatrix - first_column_submatrix <= 1) {
    return false;
  } else {
    return true;
  }
}

template class DenseDataset<float>;
template class DenseDataset<double>;
template class DenseDataset<int>;
template class DenseDataset<bool>;

} // namespace BlitzML
