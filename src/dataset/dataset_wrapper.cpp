#include <blitzml/base/common.h>
#include <blitzml/dataset/sparse_dataset.h>
#include <blitzml/dataset/sparse_binary_dataset.h>
#include <blitzml/dataset/dense_dataset.h>

namespace BlitzML {


LIBRARY_API
Dataset* BlitzML_new_sparse_dataset_float(const index_t* indices,
                                          const size_t* indptr,
                                          const float* data,
                                          index_t height, index_t width, 
                                          size_t nnz,
                                          const value_t *b, index_t length_b) {
  return new SparseDataset<float>(indices, indptr, data, 
                                  height, width, nnz, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_sparse_dataset_double(const index_t* indices,
                                           const size_t* indptr,
                                           const double* data,
                                           index_t height, index_t width, 
                                           size_t nnz,
                                           const value_t *b, index_t length_b) {
  return new SparseDataset<double>(indices, indptr, data, 
                                   height, width, nnz, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_sparse_dataset_int(const index_t* indices,
                                        const size_t* indptr,
                                        const int* data,
                                        index_t height, index_t width, 
                                        size_t nnz,
                                        const value_t *b, index_t length_b) {
  return new SparseDataset<int>(indices, indptr, data, 
                                height, width, nnz, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_sparse_dataset_bool(const index_t* indices,
                                         const size_t* indptr,
                                         const bool* data,
                                         index_t height, index_t width, 
                                         size_t nnz,
                                         const value_t *b, index_t length_b) {
  return new SparseDataset<bool>(indices, indptr, data, 
                                 height, width, nnz, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_sparse_binary_dataset(const index_t* indices,
                                           const size_t* indptr,
                                           index_t height, index_t width, 
                                           size_t nnz,
                                           const value_t *b, index_t length_b) {
  return new SparseBinaryDataset(indices, indptr, 
                                 height, width, nnz, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_dense_dataset_float(const float *data,
                                         index_t height, index_t width,
                                         const value_t *b, index_t length_b) {
  return new DenseDataset<float>(data, height, width, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_dense_dataset_double(const double *values,
                                          index_t height, index_t width,
                                          const value_t *b, index_t length_b) {
  return new DenseDataset<double>(values, height, width, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_dense_dataset_int(const int *values,
                                       index_t height, index_t width,
                                       const value_t *b, index_t length_b) {
  return new DenseDataset<int>(values, height, width, b, length_b);
}


LIBRARY_API
Dataset* BlitzML_new_dense_dataset_bool(const bool *values,
                                        index_t height, index_t width,
                                        const value_t *b, index_t length_b) {
  return new DenseDataset<bool>(values, height, width, b, length_b);
}


LIBRARY_API
value_t BlitzML_column_norm(const Dataset* data, index_t i) {
  return data->column(i)->l2_norm();
}


LIBRARY_API
value_t BlitzML_b_value_j(const Dataset* data, index_t j) {
  return data->b_value(j);
}


LIBRARY_API
void BlitzML_delete_dataset(Dataset* data) {
  delete data;
}


} // namespace BlitzML

