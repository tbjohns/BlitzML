#include <blitzml/dataset/dataset.h>

#include <cmath>

namespace BlitzML {

value_t Column::l2_norm() const {
  return sqrt(l2_norm_sq());
}


value_t Column::l2_norm_sq_centered() const {
  value_t mean_value = this->mean();
  value_t ret = l2_norm_sq() - mean_value * mean_value * length();
  if (ret <= 0) {
    return 0.;
  }
  return ret;
}


value_t Column::l2_norm_centered() const {
  return sqrt(l2_norm_sq_centered());
};


void Dataset::contiguous_submatrix_multiply(
    const std::vector<value_t> &values, value_t* result, 
    index_t first_column_submatrix, index_t end_column_submatrix) const {

  index_t j, result_ind;
  for (j = first_column_submatrix, result_ind = 0;
       j < end_column_submatrix; ++j, ++result_ind) {
    result[result_ind] = column(j)->inner_product(values);
  }
}


} // namespace BlitzML

