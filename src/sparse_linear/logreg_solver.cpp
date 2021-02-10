#include <blitzml/sparse_linear/logreg_solver.h>

#include <blitzml/base/math_util.h>
#include <blitzml/base/vector_util.h>

#include "math.h"

namespace BlitzML {

inline value_t SparseLogRegSolver::compute_prob(index_t j) {
  return is_positive_label[j] ? 1 / (1 + exp_Aomega[j])
                              : 1 - 1 / (1 + exp_Aomega[j]);
}


inline value_t SparseLogRegSolver::compute_x_value(index_t j) {
  return is_positive_label[j] ? -1 / (1 + exp_Aomega[j])
                              : 1 - 1 / (1 + exp_Aomega[j]);
}

void SparseLogRegSolver::initialize_blitz_variables(value_t* initial_conditions) {
  initialize_is_positive_label();
  initialize_model(initial_conditions);
  check_for_degenerate_problem();
  initialize_x_variables();
  check_for_poor_initialization();
  update_bias(30);
  initialize_y_and_z_variables();
}


void SparseLogRegSolver::initialize_is_positive_label() {
  is_positive_label.assign(num_examples, false);
  const value_t* labels = data->b_values();
  num_positive_labels = 0;
  for (index_t j = 0; j < num_examples; ++j) {
    if (labels[j] > 0.) {
      is_positive_label[j] = true;
      ++num_positive_labels;
    }
  }
}


void SparseLogRegSolver::initialize_x_variables() {
  compute_Aomega();

  if (l0_norm(Aomega) == 0) {
    exp_Aomega.assign(num_examples, 1);
  } else {
    exp_Aomega.resize(num_examples);
    for (index_t j = 0; j < num_examples; ++j) {
      exp_Aomega[j] = exp(Aomega[j]);
    }
  }

  x.resize(num_examples);
  ATx.resize(num_components);
  kappa_x = 1.;
  sum_x = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    x[j] = compute_x_value(j);
    sum_x += x[j];
  }
}


void SparseLogRegSolver::initialize_y_and_z_variables() {
  y.assign(num_examples, 0.);
  ATy.assign(num_components, 0.);

  z = x;
  ATz.assign(num_components, 0.);
  kappa_z = 1.0;
  z_match_x = true;
  z_match_y = false;
}


void SparseLogRegSolver::check_for_degenerate_problem() {
  problem_is_degenerate = false;
  if (use_bias) {
    if (num_positive_labels == 0) {
      problem_is_degenerate = true;
      bias = -100.;
    } else if (num_positive_labels == num_examples) {
      problem_is_degenerate = true;
      bias = 100.;
    }
    if (problem_is_degenerate) {
      omega.assign(num_components, 0.);
    }
  }
}


void SparseLogRegSolver::check_for_poor_initialization() {
  value_t max_exp_Aomega = max_vector(exp_Aomega);
  if (max_exp_Aomega > 1e30 || max_exp_Aomega != max_exp_Aomega) {
    omega.assign(num_components, 0.);
    initialize_x_variables();
  }
}


void SparseLogRegSolver::update_bias(int max_newton_itr) {
  if (!use_bias || problem_is_degenerate) {
    return;
  }

  value_t exp_delta_total = 1.0;
  if (is_vector_const(exp_Aomega)) {
    // Special case closed-form solution:
    // (this case occurs when we initialize model as all zeros)
    exp_delta_total = (exp_Aomega[0] * num_positive_labels) /
                      (num_examples - num_positive_labels);
    scale_vector(exp_Aomega, exp_delta_total);
    max_newton_itr = 0;
  }

  bool last_update_positive = false;
  for (int itr = 0; itr < max_newton_itr; ++itr) {
    // Compute derivative:
    value_t sum_p = 0.;
    value_t h = 0.;
    for (index_t j = 0; j < num_examples; ++j) {
      value_t p = 1 / (1 + exp_Aomega[j]);
      sum_p += (1 - p);
      h += p * (1 - p);
    }

    // Compute update:
    value_t deriv = num_positive_labels - sum_p;
    value_t exp_delta = 1 + deriv / h;

    if (exp_delta > 1.) {
      last_update_positive = true;
    } else if (last_update_positive) {
      break;
    } else if (exp_delta < 0.01) {
      exp_delta = num_positive_labels / sum_p;
    }

    // Apply update:
    exp_delta_total *= exp_delta;
    scale_vector(exp_Aomega, exp_delta);
  }

  value_t change = log(exp_delta_total);
  bias += change;

  sum_x = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    Aomega[j] += change;
    x[j] = compute_x_value(j);
    sum_x += x[j];
  }
  z_match_x = false;
}


void SparseLogRegSolver::perform_backtracking() {
  if (problem_is_degenerate) {
    return;
  }

  std::vector<value_t> low_exp_Aomega = exp_Aomega;

  sum_x = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    exp_Aomega[j] = exp(Aomega[j] + Delta_Aomega[j] + Delta_bias);
    x[j] = compute_x_value(j);
    sum_x += x[j];
  }
  value_t deriv_high = compute_backtracking_step_size_derivative(1.0);

  value_t step_size = 1.0;

  if (deriv_high > 0) {
    value_t high_step = 1.0;
    std::vector<value_t> high_exp_Aomega = exp_Aomega;
    value_t low_step = 0.;

    step_size = 0.5;
    int backtrack_itr = 0;
    while (++backtrack_itr) {
      sum_x = 0.;
      for (index_t j = 0; j < num_examples; ++j) {
        if (high_exp_Aomega[j] < 1e15 && high_exp_Aomega[j] > 1e-15 &&
             low_exp_Aomega[j] < 1e15 &&  low_exp_Aomega[j] > 1e-15) {
          exp_Aomega[j] = sqrt(high_exp_Aomega[j] * low_exp_Aomega[j]);
        } else {
          exp_Aomega[j] = exp(Aomega[j] +
                              step_size * (Delta_Aomega[j] + Delta_bias));
        }
        x[j] = compute_x_value(j);
        sum_x += x[j];
      }

      value_t deriv = compute_backtracking_step_size_derivative(step_size);
      if (backtrack_itr >= 5 && deriv < 0) {
        break;
      } else if (backtrack_itr >= 20) {
        break;
      }

      if (deriv < 0) {
        low_step = step_size;
        low_exp_Aomega = exp_Aomega;
      } else {
        high_step = step_size;
        high_exp_Aomega = exp_Aomega;
      }
      step_size = (high_step + low_step) / 2;
    }
  }

  for (const_index_itr ind = ws.begin_indices();
       ind != ws.end_indices();
       ++ind) {
    index_t i = ws.ith_member(*ind);
    omega[i] += step_size * Delta_omega[*ind];
  }
  bias += step_size * Delta_bias;
  for (index_t j = 0; j < num_examples; ++j) {
    Aomega[j] += step_size * (Delta_Aomega[j] + Delta_bias);
  }
}


value_t SparseLogRegSolver::compute_dual_obj() const {
  value_t loss = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    if (is_positive_label[j]) {
      if (Aomega[j] > -50) {
        loss += log1p(1/exp_Aomega[j]);
      }  else {
        loss -= Aomega[j];
      }
    } else {
      loss += log1p(exp_Aomega[j]);
    }
  }
  return -(loss + l1_penalty * l1_norm(omega));
}


void SparseLogRegSolver::update_subproblem_obj_vals() {
  obj_vals.set_dual_obj(compute_dual_obj());

  value_t gap = 0.;
  if (kappa_x >= 1) {
    kappa_x = 1;
  }
  value_t log_kappa = log1p(kappa_x - 1);
  if (log_kappa != log_kappa) {
    throw log_kappa;
  }
  for (index_t j = 0; j < num_examples; ++j) {
    value_t prob, mid_term, last_term;
    if (is_positive_label[j]) {
      prob = kappa_x / (1 + exp_Aomega[j]);
      mid_term = log1p(exp_Aomega[j] - kappa_x);
      last_term = -Aomega[j];
    } else {
      prob = kappa_x - kappa_x / (1 + exp_Aomega[j]);
      mid_term = log1p(1 / exp_Aomega[j] - kappa_x);
      last_term = Aomega[j];
    }
    gap += prob * log_kappa + (1 - prob) * mid_term + last_term;
  }

  gap += l1_penalty * l1_norm(omega);

  obj_vals.set_primal_obj_x(obj_vals.dual_obj() + gap);
}


void SparseLogRegSolver::update_newton_2nd_derivatives(value_t epsilon_to_add) {
  sum_newton_2nd_derivatives = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    value_t prob = compute_prob(j);
    newton_2nd_derivatives[j] = prob * (1 - prob) + epsilon_to_add;
    sum_newton_2nd_derivatives += newton_2nd_derivatives[j];
  }
}

} // namespace BlitzML
