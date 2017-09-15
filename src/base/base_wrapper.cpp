#include <blitzml/base/common.h>
#include <blitzml/base/solver.h>

namespace BlitzML {

LIBRARY_API
void BlitzML_solve_problem(Solver *solver, Dataset *data, Parameters *params,
                           value_t *result, char *status, const char *log_dir) {
  solver->solve(data, params, result, status, log_dir);
}


LIBRARY_API
Parameters* BlitzML_new_parameters(const value_t *values, size_t count) {
  return new Parameters(values, count);
}


LIBRARY_API
value_t BlitzML_first_parameter(const Parameters *params) {
  return (*params)[0];
}


LIBRARY_API
void BlitzML_delete_parameters(Parameters* params) {
  delete params;
}


LIBRARY_API
void BlitzML_delete_solver(Solver *solver) {
  delete solver;
}


LIBRARY_API
void BlitzML_set_tolerance(Solver *solver, value_t value) {
  solver->set_tolerance(value);
}


LIBRARY_API
value_t BlitzML_tolerance(Solver *solver) {
  return solver->tolerance();
}


LIBRARY_API
void BlitzML_set_max_time(Solver *solver, value_t value) {
  solver->set_max_time(value);
}


LIBRARY_API
value_t BlitzML_max_time(Solver *solver) {
  return solver->max_time();
}


LIBRARY_API
void BlitzML_set_max_iterations(Solver *solver, unsigned value) {
  solver->set_max_iterations(value);
}


LIBRARY_API
unsigned BlitzML_max_iterations(Solver *solver) {
  return solver->max_iterations();
}


LIBRARY_API
void BlitzML_set_min_time(Solver *solver, value_t value) {
  solver->set_min_time(value);
}


LIBRARY_API
value_t BlitzML_min_time(Solver *solver) {
  return solver->min_time();
}


LIBRARY_API
void BlitzML_set_verbose(Solver *solver, bool value) {
  solver->set_verbose(value);
}


LIBRARY_API
bool BlitzML_verbose(Solver *solver) {
  return solver->verbose();
}


LIBRARY_API
void BlitzML_set_use_screening(Solver *solver, bool value) {
  solver->set_use_screening(value);
}


LIBRARY_API
bool BlitzML_use_screening(Solver *solver) {
  return solver->use_screening();
}


LIBRARY_API
void BlitzML_set_use_working_sets(Solver *solver, bool value) {
  solver->set_use_working_sets(value);
}


LIBRARY_API
bool BlitzML_use_working_sets(Solver *solver) {
  return solver->use_working_sets();
}


LIBRARY_API
void BlitzML_set_log_vectors(Solver *solver, bool value) {
  solver->set_log_vectors(value);
}


LIBRARY_API
bool BlitzML_log_vectors(Solver *solver) {
  return solver->log_vectors();
}


LIBRARY_API
void BlitzML_set_suppress_warnings(Solver *solver, bool value) {
  solver->set_suppress_warnings(value);
}


LIBRARY_API
bool BlitzML_suppress_warnings(Solver *solver) {
  return solver->suppress_warnings();
}


} // namespace BlitzML
