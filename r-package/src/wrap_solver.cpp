#include <Rcpp.h>

#include <blitzml/base/common.h>
#include <blitzml/dataset/dataset.h>
#include <blitzml/sparse_linear/sparse_linear_solver.h>
#include <blitzml/sparse_linear/lasso_solver.h>
#include <blitzml/sparse_linear/logreg_solver.h>
#include <blitzml/dataset/sparse_dataset.h>


// [[Rcpp::export]]
SEXP BlitzML_new_solver() {
  BlitzML::SparseLinearSolver* solver = new BlitzML::SparseLinearSolver();
  Rcpp::XPtr<BlitzML::SparseLinearSolver> ptr(solver, true);
  return ptr;
}

// [[Rcpp::export]]
SEXP BlitzML_new_logreg_solver() {
  BlitzML::SparseLogRegSolver* solver = new BlitzML::SparseLogRegSolver();
  Rcpp::XPtr<BlitzML::SparseLogRegSolver> ptr(solver, true);
  return ptr;
}

// [[Rcpp::export]]
SEXP BlitzML_new_linear_solver() {
  BlitzML::LassoSolver* solver = new BlitzML::LassoSolver();
  Rcpp::XPtr<BlitzML::LassoSolver> ptr(solver, true);
  return ptr;
}

// [[Rcpp::export]]
SEXP BlitzML_new_parameters(const Rcpp::NumericVector params) {
  BlitzML::Parameters* blitz_params = new BlitzML::Parameters((double*)params.begin(), params.size() );
  Rcpp::XPtr<BlitzML::Parameters> ptr(blitz_params, true);
  return ptr;
}

// [[Rcpp::export]]
double BlitzML_solver_compute_max_l1_penalty(
    SEXP xptr_solver,
    SEXP xptr_dataset,
    SEXP xptr_params) {
  Rcpp::XPtr< BlitzML::SparseLinearSolver > solver(xptr_solver);
  Rcpp::XPtr< BlitzML::Dataset> data(xptr_dataset);
  Rcpp::XPtr< BlitzML::Parameters > params(xptr_params);
  return solver->compute_max_l1_penalty(data, params);
}

// [[Rcpp::export]]
void BlitzML_solve_problem(SEXP xptr_solver,
                           SEXP xptr_dataset,
                           SEXP xptr_params,
                           Rcpp::NumericVector &result,
                           Rcpp::RawVector &status_buffer,
                           const Rcpp::String &log_dir) {
  Rcpp::XPtr< BlitzML::SparseLinearSolver > solver(xptr_solver);
  Rcpp::XPtr< BlitzML::SparseDataset<double> > data(xptr_dataset);
  Rcpp::XPtr< BlitzML::Parameters > params(xptr_params);
  double *result_ptr = (double *)result.begin();
  char *buf = (char *)status_buffer.begin();
  solver->solve(data, params, result_ptr, buf, log_dir.get_cstring());
}

// [[Rcpp::export]]
void BlitzML_set_tolerance(SEXP xptr_solver, double value) {
  Rcpp::XPtr< BlitzML::SparseLinearSolver > solver(xptr_solver);
  solver->set_tolerance(value);
}
// [[Rcpp::export]]
void BlitzML_set_max_time(SEXP xptr_solver, double value) {
  Rcpp::XPtr< BlitzML::SparseLinearSolver > solver(xptr_solver);
  solver->set_max_time(value);
}

// [[Rcpp::export]]
void BlitzML_set_max_iterations(SEXP xptr_solver, unsigned value) {
  Rcpp::XPtr< BlitzML::SparseLinearSolver > solver(xptr_solver);
  solver->set_max_iterations(value);
}

// [[Rcpp::export]]
void BlitzML_set_use_working_sets(SEXP xptr_solver, unsigned value) {
  Rcpp::XPtr< BlitzML::SparseLinearSolver > solver(xptr_solver);
  solver->set_use_working_sets(value);
}
