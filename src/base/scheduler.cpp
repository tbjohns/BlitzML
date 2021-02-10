#include <blitzml/base/scheduler.h>
#include <blitzml/base/math_util.h>
#include <blitzml/base/capsule.h>

#include <cmath>
#include <algorithm>
#include <iostream>

namespace BlitzML {

void Scheduler::set_epsilon_candidates(const std::vector<value_t> &epsilons) {
  epsilon_candidates = epsilons;
}


void Scheduler::capsule_candidates(std::vector<CapsuleParams>& capsule_candidates) const {

  capsule_candidates.resize(xi_candidates.size());
  CapsuleCalculator cap_calc(gamma, current_Delta, current_dsq);
  for (size_t ind = 0; ind < xi_candidates.size(); ++ind) {
    cap_calc.compute_capsule_params(xi_candidates[ind], capsule_candidates[ind]);
  }
}


value_t Scheduler::estimate_C_progress() {
  value_t ret = 1.;
  if (C_progress_estimates.size() > 0) {
    ret = median_last_k(C_progress_estimates, 2);
  }
  if (ret == ret && ret >= 1.) {
    return ret;
  } else {
    return 1.;
  }
  return 1.0;
}


void Scheduler::set_problem_sizes(const std::vector<value_t> &problem_sizes) {
  problem_size_candidates = problem_sizes;
}


void Scheduler::optimize_xi_and_epsilon(unsigned short &best_capsule_ind,
                                         value_t &best_epsilon,
                                         double &best_time_limit) {

  assert_with_error_message(gamma > 0, "in scheduler, must set gamma to be > 0");
  assert_with_error_message(xi_candidates.size() > 0,
         "in scheduler, must have at least 1 xi candidate");
  assert_with_error_message(epsilon_candidates.size() > 0,
         "in scheduler, must have at least 1 epsilon candidate");

  value_t C_progress = estimate_C_progress();

  double T_overhead = overhead_time_estimate();
  value_t C_solve;
  if (C_solve_time_estimates.size() > 0) {
    C_solve = median_last_k(C_solve_time_estimates, 5);
  } else {
    C_solve = 1.;
  }

  value_t best_value = 1e20;
  best_capsule_ind = 0;
  best_epsilon = 0.1;
  best_time_limit = 20 * T_overhead;

  for (size_t ind = 0; ind < xi_candidates.size(); ++ind) {
    value_t xi = xi_candidates[ind];
    for (size_t eps_ind = 0; eps_ind < epsilon_candidates.size(); ++eps_ind) {
      value_t epsilon = epsilon_candidates[eps_ind];
      value_t problem_size = problem_size_candidates[ind];

      value_t next_Delta_est = (1 - (1 - epsilon) * xi * C_progress) * current_Delta;
      value_t min_next_Delta_est = epsilon * current_Delta;
      if (next_Delta_est < min_next_Delta_est) {
        next_Delta_est = min_next_Delta_est;
      }

      value_t numerator = -log(current_Delta / next_Delta_est);

      value_t epsilon_term = subproblem_time_complexity(epsilon);
      value_t T_solve = C_solve * problem_size * epsilon_term;
      value_t denominator = T_overhead + T_solve;

      value_t value = numerator / denominator;

      if (value < best_value) {
        best_value = value;
        best_capsule_ind = ind;
        best_epsilon = epsilon;
        best_time_limit = T_solve;
      }
    }
  }
}


void Scheduler::record_subproblem_size(value_t xi,
                                       value_t subproblem_size) {
  chosen_xi = xi;
  chosen_subproblem_size = subproblem_size;
}


void Scheduler::record_overhead_time(double time) {
  C_overhead_time_estimates.push_back(time);
}


void Scheduler::record_subproblem_progress(value_t new_Delta,
                                           value_t subproblem_Delta,
                                           double subproblem_time,
                                           double alpha) {
  value_t realized_epsilon = subproblem_Delta / current_Delta;
  const double max_realized_epsilon = 0.99;
  if (realized_epsilon > max_realized_epsilon) {
    realized_epsilon = max_realized_epsilon;
  }

  value_t epsilon_term = subproblem_time_complexity(realized_epsilon);
  value_t C_solve_time_est = subproblem_time / chosen_subproblem_size / epsilon_term;
  C_solve_time_estimates.push_back(C_solve_time_est);

  if (realized_epsilon == max_realized_epsilon) {
    return;
  }

  value_t C_progress_est = 1;
  //if (alpha == 1) {
    //C_progress_est = 1.5 / chosen_xi;
  //} else {
    C_progress_est = (1 - new_Delta / current_Delta) / (1 - realized_epsilon) / chosen_xi;
  //}

  /*
  if (C_progress_est != C_progress_est || C_progress_est < 1) {
    C_progress_est = 1;
  }
  */

  C_progress_estimates.push_back(C_progress_est);
}


value_t Scheduler::subproblem_time_complexity(value_t epsilon) {
  return 1 / epsilon;
}


double Scheduler::overhead_time_estimate() {
  assert_with_error_message(C_overhead_time_estimates.size() > 0,
         "cannot estimate overhead time before calling record_overhead_time");
  return median_last_k(C_overhead_time_estimates, 5);
}


void Scheduler::reset() {
  epsilon_candidates.clear();
  epsilon_candidates.push_back(0.01);
  epsilon_candidates.push_back(0.02);
  epsilon_candidates.push_back(0.03);
  epsilon_candidates.push_back(0.05);
  epsilon_candidates.push_back(0.07);
  epsilon_candidates.push_back(0.1);
  epsilon_candidates.push_back(0.2);
  epsilon_candidates.push_back(0.3);
  epsilon_candidates.push_back(0.5);
  epsilon_candidates.push_back(0.7);

  //const int num_xi_values = 1;
  //value_t xi_values[num_xi_values] = { 0.1 };

	const int num_xi_values = 125;
  value_t xi_values[num_xi_values] = {
				 1.00000000e-06,   1.46779927e-06,   2.15443469e-06,
         3.16227766e-06,   4.64158883e-06,   6.81292069e-06,
         1.00000000e-05,   1.46779927e-05,   2.15443469e-05,
         3.16227766e-05,   4.64158883e-05,   6.81292069e-05,
         1.00000000e-04,   1.46779927e-04,   2.15443469e-04,
         3.16227766e-04,   4.64158883e-04,   6.81292069e-04,
         1.00000000e-03,   1.46779927e-03,   2.15443469e-03,
         3.16227766e-03,   4.64158883e-03,   6.81292069e-03,
         1.00000000e-02,   1.00000000e-02,   1.04761575e-02,
         1.09749877e-02,   1.14975700e-02,   1.20450354e-02,
         1.26185688e-02,   1.32194115e-02,   1.38488637e-02,
         1.45082878e-02,   1.51991108e-02,   1.59228279e-02,
         1.66810054e-02,   1.74752840e-02,   1.83073828e-02,
         1.91791026e-02,   2.00923300e-02,   2.10490414e-02,
         2.20513074e-02,   2.31012970e-02,   2.42012826e-02,
         2.53536449e-02,   2.65608778e-02,   2.78255940e-02,
         2.91505306e-02,   3.05385551e-02,   3.19926714e-02,
         3.35160265e-02,   3.51119173e-02,   3.67837977e-02,
         3.85352859e-02,   4.03701726e-02,   4.22924287e-02,
         4.43062146e-02,   4.64158883e-02,   4.86260158e-02,
         5.09413801e-02,   5.33669923e-02,   5.59081018e-02,
         5.85702082e-02,   6.13590727e-02,   6.42807312e-02,
         6.73415066e-02,   7.05480231e-02,   7.39072203e-02,
         7.74263683e-02,   8.11130831e-02,   8.49753436e-02,
         8.90215085e-02,   9.32603347e-02,   9.77009957e-02,
         1.02353102e-01,   1.07226722e-01,   1.12332403e-01,
         1.17681195e-01,   1.23284674e-01,   1.29154967e-01,
         1.35304777e-01,   1.41747416e-01,   1.48496826e-01,
         1.55567614e-01,   1.62975083e-01,   1.70735265e-01,
         1.78864953e-01,   1.87381742e-01,   1.96304065e-01,
         2.05651231e-01,   2.15443469e-01,   2.25701972e-01,
         2.36448941e-01,   2.47707636e-01,   2.59502421e-01,
         2.71858824e-01,   2.84803587e-01,   2.98364724e-01,
         3.12571585e-01,   3.27454916e-01,   3.43046929e-01,
         3.59381366e-01,   3.76493581e-01,   3.94420606e-01,
         4.13201240e-01,   4.32876128e-01,   4.53487851e-01,
         4.75081016e-01,   4.97702356e-01,   5.21400829e-01,
         5.46227722e-01,   5.72236766e-01,   5.99484250e-01,
         6.28029144e-01,   6.57933225e-01,   6.89261210e-01,
         7.22080902e-01,   7.56463328e-01,   7.92482898e-01,
         8.30217568e-01,   8.69749003e-01,   9.11162756e-01,
         9.54548457e-01,   1.00000000e+00 };

  xi_candidates.assign(xi_values, xi_values + num_xi_values);
  std::sort(xi_candidates.begin(), xi_candidates.end());

  C_progress_estimates.clear();
  C_solve_time_estimates.clear();
  C_overhead_time_estimates.clear();

  gamma = -1.0;
}

}

