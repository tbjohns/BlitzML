#include <Rcpp.h>

#include <blitzml/base/common.h>
#include <blitzml/dataset/dataset.h>
#include <blitzml/sparse_linear/sparse_linear_solver.h>
#include <blitzml/sparse_linear/lasso_solver.h>
#include <blitzml/sparse_linear/logreg_solver.h>
#include <blitzml/dataset/sparse_dataset.h>

using namespace BlitzML;

// [[Rcpp::export]]
SEXP BlitzML_new_solver() {
  BlitzML::SparseLogRegSolver* solver = new BlitzML::SparseLogRegSolver();
  Rcpp::XPtr<BlitzML::SparseLogRegSolver> ptr(solver, true);
  return ptr;
}

// [[Rcpp::export]]
SEXP BlitzML_new_parameters(const Rcpp::NumericVector params) {
  BlitzML::Parameters* blitz_params = new BlitzML::Parameters((double*)params.begin(), params.size() );
  Rcpp::XPtr<BlitzML::Parameters> ptr(blitz_params, true);
  return ptr;
}

// [[Rcpp::export]]
double BlitzML_sparse_linear_solver_compute_max_l1_penalty(
    SEXP xptr_solver,
    SEXP xptr_dataset,
    SEXP xptr_params) {
  Rcpp::XPtr< BlitzML::SparseLogRegSolver > solver(xptr_solver);
  Rcpp::XPtr< BlitzML::SparseDataset<double> > data(xptr_dataset);
  Rcpp::XPtr< BlitzML::Parameters > params(xptr_params);
  return solver->compute_max_l1_penalty(data, params);
}
