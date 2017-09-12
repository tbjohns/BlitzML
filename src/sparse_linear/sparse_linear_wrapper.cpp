#include <blitzml/sparse_linear/sparse_linear_solver.h> 
#include <blitzml/sparse_linear/lasso_solver.h> 
#include <blitzml/sparse_linear/logreg_solver.h> 


namespace BlitzML {

LIBRARY_API
Solver* BlitzML_new_sparse_linear_solver() {
  return new SparseLinearSolver();
}


LIBRARY_API
Solver* BlitzML_new_lasso_solver() {
  return new LassoSolver();
}


LIBRARY_API
Solver* BlitzML_new_sparse_logreg_solver() {
  return new SparseLogRegSolver();
}


LIBRARY_API
value_t BlitzML_sparse_linear_solver_compute_max_l1_penalty(
    SparseLinearSolver* solver, const Dataset* data, const Parameters* params) {
  return solver->compute_max_l1_penalty(data, params);
}


}
