#include <Rcpp.h>

template< typename T>
class MappedCSC {
public:
  MappedCSC();
  MappedCSC(std::uint32_t n_rows,
            std::uint32_t n_cols,
            uint32_t nnz,
            std::uint32_t * row_indices,
            std::uint32_t * col_ptrs,
            T * values):
    n_rows(n_rows), n_cols(n_cols), nnz(nnz), row_indices(row_indices), col_ptrs(col_ptrs), values(values) {};
  const std::uint32_t n_rows;
  const std::uint32_t n_cols;
  const uint32_t nnz;
  const std::uint32_t * row_indices;
  const std::uint32_t * col_ptrs;
  T * values;
};

using dMappedCSC = MappedCSC<double>;

dMappedCSC extract_mapped_csc(Rcpp::S4 input) {
  Rcpp::IntegerVector dim = input.slot("Dim");
  Rcpp::NumericVector values = input.slot("x");
  uint32_t nrows = dim[0];
  uint32_t ncols = dim[1];
  Rcpp::IntegerVector row_indices = input.slot("i");
  Rcpp::IntegerVector col_ptrs = input.slot("p");
  return dMappedCSC(nrows, ncols, values.length(), (uint32_t *)row_indices.begin(), (uint32_t *)col_ptrs.begin(), values.begin());
}
