#include <blitzml/base/solver.h>
#include <blitzml/base/vector_util.h>
#include <blitzml/base/math_util.h>

#include <iostream>
#include <limits>
#include <algorithm>
#include <utility>
#include <string.h>
#include <cmath>

using std::vector;
using std::swap;
using std::max; 
using std::min;


namespace BlitzML {

void Solver::solve(const Dataset *data, const Parameters *params, 
                   value_t *result, char *status_message,
                   const char *log_directory) {
  solver_timer.reset();
  solver_logger.set_suppress_warnings(suppress_warnings());
  solver_logger.set_log_directory(log_directory);
  solver_logger.throttle_logging_with_interval(max_time() / 100);

  this->data = data;
  this->params = params;

  num_components = initial_problem_size();

  ws.set_max_size(num_components);
  ws.clear();

  initialize_priority_vectors();
  initialize(result);
  scheduler.reset();
  scheduler.set_strong_convexity_constant(strong_convexity_constant()); 

  obj_vals.set_dual_obj(compute_dual_obj());
  obj_vals.set_primal_obj_y(1e30);

  iteration_number = 0;
  while (++iteration_number) {
    Timer overhead_timer;
    update_alpha();
    update_y();
    update_obj_vals();
    log_info(should_force_log());
    record_subproblem_progress();
    if (use_screening()) {
      screen_components();
    }
    scheduler.record_overhead_time(overhead_timer.elapsed_time());
    
    setup_subproblem();

    subproblem_solve_timer.reset();
    solve_subproblem();
    subproblem_solve_timer.pause_timing();

    Solver::ConvergenceStatus convergence_status = get_convergence_status();
    if (convergence_status != NOT_CONVERGED) {
      ++iteration_number;
      log_info(true);
      set_status_message(status_message, convergence_status);
      fill_result(result);
      solver_logger.close_files();
      return;
    }
  }
}


void Solver::initialize_priority_vectors() {
  priority_values.resize(num_components);
}


void Solver::update_alpha() {
  if (!use_working_sets() && iteration_number > 1) {
    alpha = 1.;
  } else {
    alpha = compute_alpha();
  }
  debug("alpha is %0.15f", alpha);
}


void Solver::update_obj_vals() {
  duality_gap_last_iteration = obj_vals.duality_gap();
  dual_objective_last_iteration = obj_vals.dual_obj();

  value_t primal_obj_y;
  if (alpha == 1. && iteration_number > 1) {
    primal_obj_y = obj_vals.primal_obj_z();
  } else {
    primal_obj_y = compute_primal_obj_y();
  }
  obj_vals.set_primal_obj_y(primal_obj_y);
}


void Solver::record_subproblem_progress() {
  if (iteration_number == 1) {
    return;
  }

  value_t subproblem_time = subproblem_solve_timer.elapsed_time();
  value_t subproblem_duality_gap = obj_vals.primal_obj_x() - obj_vals.dual_obj();
  scheduler.record_subproblem_progress(obj_vals.duality_gap(),
                                       subproblem_duality_gap,
                                       subproblem_time, 
                                       alpha);
}


value_t Solver::strong_convexity_constant() const {
  return 1;
}


void Solver::setup_subproblem() {
  if (use_working_sets()) {
    set_subproblem_parameters();
    record_subproblem_size();
  } else {
    setup_subproblem_no_working_sets();
  }
}


void Solver::setup_subproblem_no_working_sets() {
  if (iteration_number == 1) {
    ws.clear();
    for (index_t i = 0; i < num_components; ++i) {
      ws.add_index(i);
    }
  }
  subproblem_params.epsilon = 0.01;
  subproblem_params.max_iterations = 5;
  subproblem_params.min_time = 0.;
  subproblem_params.time_limit = max_time();
}


void Solver::update_priority_values() {
  int total = 0;
  for (index_t i = 0; i < num_components; ++i) {
    unsigned short capsule_ind = compute_priority_value(i);
    priority_values[i] = capsule_ind;
    if (capsule_ind < capsule_candidates.size()) {
      ++total;
      problem_size_diffs[capsule_ind] += size_of_component(i);
    } 
  }
}


void Solver::set_initial_subproblem_state(SubproblemState& initial_state) const {
  initial_state.dual_obj = obj_vals.dual_obj();
  initial_state.x = x;
}


bool Solver::check_sufficient_dual_progress(
        const SubproblemState& initial_state, value_t epsilon) const {
  value_t dual_obj_progress = obj_vals.dual_obj() - initial_state.dual_obj;
  value_t norm_diff_sq = norm_diff_sq_z_initial_x(initial_state);
  value_t min_dual_progress = 0.5 * strong_convexity_constant() * (1 - epsilon) * norm_diff_sq;
  return (dual_obj_progress >= min_dual_progress);
}


bool Solver::should_force_log() {
  if (iteration_number == 1) {
    return true;
  }
  if (obj_vals.duality_gap() < 0.2 * last_log_gap) {
    return true;
  }
  return false;
}


void Solver::log_info(bool force_log) {
  solver_timer.pause_timing();

  double elapsed_time = solver_timer.elapsed_time();
  if (verbose()) {
    print("Time: %5e, dual objective: %7e, duality gap: %7e", 
           elapsed_time, obj_vals.dual_obj(), obj_vals.duality_gap());
  }

  bool logged = solver_logger.log_new_point(elapsed_time, obj_vals.dual_obj(), force_log);
  if (logged) {
    solver_logger.log_value<value_t>("primal_obj_y", obj_vals.primal_obj_y());
    solver_logger.log_value<unsigned>("iteration", iteration_number);
    solver_logger.log_value<value_t>("alpha", alpha);
    solver_logger.log_value<value_t>("xi", subproblem_params.xi);
    solver_logger.log_value<value_t>("epsilon", subproblem_params.epsilon);
    solver_logger.log_value<size_t>("working_set_size", ws.size());
    solver_logger.log_value<index_t>("num_not_eliminated", num_components);
    if (log_vectors()) {
      solver_logger.log_vector<value_t>("y", y);
      solver_logger.log_vector<value_t>("x", x);
    }
    log_variables(solver_logger);
    last_log_gap = obj_vals.duality_gap();
  }

  solver_timer.continue_timing();
}


Solver::ConvergenceStatus Solver::get_convergence_status() {
  double elapsed_time = solver_timer.elapsed_time();
  if (obj_vals.duality_gap() == 0. && (elapsed_time > min_time())) {
    return REACHED_STOPPING_TOLERANCE;
  } else if (obj_vals.dual_obj() != 0. 
             && obj_vals.duality_gap() / fabs(obj_vals.dual_obj()) < tolerance()
             && elapsed_time > min_time()) {
    return REACHED_STOPPING_TOLERANCE;
  }
  if (elapsed_time > max_time()) {
    return EXCEEDED_TIME_LIMIT;
  }
  if (iteration_number > max_iterations()) {
    return EXCEEDED_MAX_ITERATIONS;
  }
  if (iteration_number > 1 
      && obj_vals.dual_obj() <= dual_objective_last_iteration
      && obj_vals.duality_gap() >= duality_gap_last_iteration) {
    return REACHED_MACHINE_PRECISION;
  }

  return NOT_CONVERGED;
}


void Solver::set_status_message(char* status_message, 
                                Solver::ConvergenceStatus status) {
  switch (status) {
    case REACHED_STOPPING_TOLERANCE: 
      strcpy(status_message, "Reached stopping tolerance");
      break;
    case EXCEEDED_TIME_LIMIT:
      strcpy(status_message, "Exceeded time limit");
      break;
    case EXCEEDED_MAX_ITERATIONS:
      strcpy(status_message, "Exceeded iterations limit");
      break;
    case REACHED_MACHINE_PRECISION:
      strcpy(status_message, "Reached machine precision");
      break;
    default:
      strcpy(status_message, "Convergence status unknown");
      break;
  }
}


value_t Solver::compute_d_sq() const {
  return l2_norm_diff_sq(x, y);
}


void Solver::set_subproblem_parameters() {
  scheduler.set_current_duality_gap(obj_vals.duality_gap());
  scheduler.set_current_dsq(compute_d_sq());

  scheduler.capsule_candidates(capsule_candidates);
  size_t num_capsule_candidates = capsule_candidates.size();

  problem_size_diffs.assign(num_capsule_candidates, 0.);
  update_priority_values();

  vector<value_t> problem_size_candidates(num_capsule_candidates, 0);
  problem_size_candidates[0] = problem_size_diffs[0];
  for (size_t ind = 1; ind < num_capsule_candidates; ++ind) {
    problem_size_candidates[ind] = problem_size_candidates[ind-1] + problem_size_diffs[ind];
  }

  double overhead_time_est = scheduler.overhead_time_estimate();
  subproblem_params.max_iterations = 1000;
  subproblem_params.min_time = 0.1 * overhead_time_est; 

  value_t best_epsilon;
  double best_time_limit;
  unsigned short best_capsule_ind;
  if (iteration_number > 1) {
    scheduler.set_problem_sizes(problem_size_candidates);
    scheduler.optimize_xi_and_epsilon(best_capsule_ind, best_epsilon, best_time_limit);
  } else {
    best_epsilon = 0.5;
    best_time_limit = 2 * overhead_time_est; 
    best_capsule_ind = 0;
    while ((capsule_candidates[best_capsule_ind].xi < 1.0) && 
           (problem_size_candidates[best_capsule_ind] < problem_size_candidates.back())) {
      ++best_capsule_ind;
    }
    subproblem_params.max_iterations = 1;
  }
  subproblem_params.epsilon = best_epsilon;

  double max_time_limit = overhead_time_est * 15;
  double min_time_limit = overhead_time_est * 0.5;
  subproblem_params.time_limit = max(min(best_time_limit, max_time_limit), min_time_limit);
  debug("time ratio is %f", subproblem_params.time_limit / overhead_time_est);
  debug("epsilon is %f", subproblem_params.epsilon);

  subproblem_params.xi = capsule_candidates[best_capsule_ind].xi;

  ws.clear();
  for (index_t i = 0; i < num_components; ++i) {
    if (priority_values[i] <= best_capsule_ind) {
      ws.add_index(i);
    }
  }
  debug("size working set is %d", ws.size());
}


void Solver::record_subproblem_size() {
  size_t size_subproblem = 0;
  for (const_index_itr i = ws.begin_sorted(); i != ws.end_sorted(); ++i) {
    size_subproblem += size_of_component(*i);
  }
  scheduler.record_subproblem_size(subproblem_params.xi, static_cast<value_t>(size_subproblem));
}

} // namespace BlitzML

