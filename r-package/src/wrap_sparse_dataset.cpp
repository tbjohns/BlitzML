#include <Rcpp.h>
#include "mapped_csc_matrix.h"
#include <blitzml/dataset/sparse_dataset.h>


// [[Rcpp::export]]
SEXP BlitzML_new_sparse_dataset(const Rcpp::S4 &x,
                                       const Rcpp::NumericVector &y) {
  dMappedCSC x_csc = extract_mapped_csc(x);
  double *y_ptr = (double *)y.begin();
  BlitzML::SparseDataset<double>* dataset =
    new BlitzML::SparseDataset<double>(x_csc.row_indices, x_csc.col_ptrs, x_csc.values,
                                       x_csc.n_rows, x_csc.n_cols, x_csc.nnz, y_ptr, y.size());
  Rcpp::XPtr<BlitzML::SparseDataset<double>> ptr(dataset, true);
  return ptr;
}
