#include <blitzml/sparse_linear/logreg_solver.h>

#include <blitzml/base/math_util.h>
#include <blitzml/base/vector_util.h>

#include "math.h"

namespace BlitzML {


void SparseLogRegSolver::initialize_blitz_variables(value_t* initial_conditions) {

  is_positive_label.assign(num_examples, false);
  const value_t* labels = data->b_values();
  index_t num_positive = 0;
  for (index_t j = 0; j < num_examples; ++j) {
    if (labels[j] >= 0.) {
      is_positive_label[j] = true;
      ++num_positive;
    }
  }

  x.resize(num_examples);
  ATx.resize(num_components);
  kappa_x = 1.;

  y.assign(num_examples, 0.);
  ATy.assign(num_components, 0.);

  if (initial_conditions != NULL) {
    omega.assign(initial_conditions, initial_conditions + num_components);
  } else {
    omega.assign(num_components, 0.);
  }
  bias = 0.;

  problem_is_degenerate = false;
  if (use_bias) {
    if (num_positive == 0) {
      problem_is_degenerate = true;
      bias = -100.;
    } else if (num_positive == num_examples) {
      problem_is_degenerate = true;
      bias = 100.;
    }
    if (problem_is_degenerate) {
      omega.assign(num_components, 0.);
    }
  }

  initialize_x_variables();
  value_t max_exp_bAomega = max_vector(exp_bAomega);
  if (max_exp_bAomega > 1e30) {
    omega.assign(num_components, 0.);
    initialize_x_variables();
  }

  update_bias(25);

  z = x;
  ATz.assign(num_components, 0.);
  kappa_z = 1.0;
  z_match_x = true;
  z_match_y = false;

  screen_indices_map.resize(num_components);
  for (index_t i = 0; i < num_components; ++i) {
    screen_indices_map[i] = i;
  }
}

void SparseLogRegSolver::initialize_x_variables() {
  compute_Aomega();
  if (l0_norm(Aomega) == 0) {
    exp_bAomega.assign(num_examples, 1);
  } else {
    exp_bAomega.resize(num_examples);
    const value_t* labels = data->b_values();
    for (index_t i = 0; i < num_examples; ++i) {
      exp_bAomega[i] = exp(labels[i] * Aomega[i]);
    }
  }
  for (index_t i = 0; i < num_examples; ++i) {
    x[i] = -data->b_value(i) * (1 / (1 + exp_bAomega[i]));
  }
  sum_x = sum_vector(x);
}


void SparseLogRegSolver::update_bias(int max_newton_itr) {
  if (!use_bias || problem_is_degenerate) {
    return;
  }

  // Convert exp_bAomega to exp_minus_Aomega:
  value_t pos_count = 0;
  for (index_t j = 0; j < num_examples; ++j) {
    if (is_positive_label[j]) {
      exp_bAomega[j] = 1 / exp_bAomega[j];
      if (exp_bAomega[j] > 1e30 || exp_bAomega[j] != exp_bAomega[j]) {
        exp_bAomega[j] = exp(-Aomega[j]);
      }
      pos_count += 1.0;
    }
  }

  // Special case closed-form solution:
  // (this case occurs when we initialize model as all zeros)
  value_t exp_delta_total = 1.0;  
  if (is_vector_const(exp_bAomega, 1e-8)) {
    exp_delta_total = pos_count / (num_examples - pos_count) / exp_bAomega[0];
    scale_vector(exp_bAomega, 1/exp_delta_total);
    max_newton_itr = 0;
  }

  int sign_last_deriv = 0;
  for (int itr = 0; itr < max_newton_itr; ++itr) {
    // Compute derivative:
    value_t sum_p = 0.;
    value_t h = 0.;
    for (index_t j = 0; j < num_examples; ++j) {
      value_t p = 1.0 / (1 + exp_bAomega[j]);
      sum_p += p;
      h += p * (1 - p);
    }

    // Check stopping condition
    value_t deriv = pos_count - sum_p;
    if (deriv == 0.) {
      break;
    } else {
      sign_last_deriv = sign(deriv);
    }

    // Choose update:
    value_t exp_delta = 1 + deriv / h;
    if (exp_delta < 0.1) {
      exp_delta = pos_count / sum_p;
    }

    // Apply update:
    exp_delta_total *= exp_delta;
    value_t exp_delta_inv = 1 / exp_delta;
    for (index_t j = 0; j < num_examples; ++j) {
      exp_bAomega[j] *= exp_delta_inv;
    }
  }

  value_t change = log(exp_delta_total);
  bias += change;
  
  sum_x = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    Aomega[j] += change;
    if (is_positive_label[j]) {
      exp_bAomega[j] = 1 / exp_bAomega[j];
      if (exp_bAomega[j] > 1e30 || exp_bAomega[j] != exp_bAomega[j]) {
        exp_bAomega[j] = exp(Aomega[j]);
      }
      x[j] = -(1 / (1 + exp_bAomega[j]));
    } else {
      if (exp_bAomega[j] > 1e30 || exp_bAomega[j] != exp_bAomega[j]) {
        exp_bAomega[j] = exp(-Aomega[j]);
      }
      x[j] = (1 / (1 + exp_bAomega[j]));
    }
    sum_x += x[j];
  }
  z_match_x = false;
}


void SparseLogRegSolver::perform_backtracking() {
  if (problem_is_degenerate) {
    return;
  }

  std::vector<value_t> low_exp_bAomega = exp_bAomega;

  sum_x = 0.;
  for (int i = 0; i < num_examples; ++i) {
    exp_bAomega[i] = exp(data->b_value(i) * (Aomega[i] + Delta_Aomega[i] + Delta_bias));
    x[i] = -data->b_value(i) / (1 + exp_bAomega[i]);
    sum_x += x[i];
  }
  value_t deriv_high = compute_backtracking_step_size_derivative(1.0);

  value_t step_size = 1.0;

  if (deriv_high > 0) {
    value_t high_step = 1.0;
    std::vector<value_t> high_exp_bAomega = exp_bAomega;
    value_t low_step = 0.;

    step_size = 0.5;
    int backtrack_itr = 0;
    const value_t* labels = data->b_values();
    while (++backtrack_itr) {
      sum_x = 0.;
      for (index_t i = 0; i < num_examples; ++i) {
        if (high_exp_bAomega[i] < 1e10 && high_exp_bAomega[i] > 1e-10 &&
             low_exp_bAomega[i] < 1e10 &&   low_exp_bAomega[i] > 1e-10) {
          exp_bAomega[i] = sqrt(high_exp_bAomega[i] * low_exp_bAomega[i]);
        } else {
          exp_bAomega[i] = exp(labels[i] * (Aomega[i] + step_size * (Delta_Aomega[i] + Delta_bias)));
        }
        x[i] = -data->b_value(i) / (1 + exp_bAomega[i]);
        sum_x += x[i];
      }

      value_t deriv = compute_backtracking_step_size_derivative(step_size);
      if (backtrack_itr >= 5 && deriv < 0) {
        break;
      } else if (backtrack_itr >= 15) {
        break;
      }

      if (deriv < 0) {
        low_step = step_size;
        low_exp_bAomega = exp_bAomega;
      } else {
        high_step = step_size;
        high_exp_bAomega = exp_bAomega;
      }
      step_size = (high_step + low_step) / 2;
    }
  }

  for (const_index_itr ind = ws.begin_indices(); ind != ws.end_indices(); ++ind) {
    index_t i = ws.ith_member(*ind);
    omega[i] += step_size * Delta_omega[*ind];
  }
  Delta_bias *= step_size;
  bias += Delta_bias;
  for (index_t i = 0; i < num_examples; ++i) {
    Aomega[i] += step_size * Delta_Aomega[i] + Delta_bias;
  }
}


value_t SparseLogRegSolver::compute_dual_obj() const {
  value_t loss = 0.;
  for (index_t i = 0; i < num_examples; ++i) {
    loss += log1p(1/exp_bAomega[i]);
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
  const value_t* labels = data->b_values();
  for (int i = 0; i < num_examples; ++i) {
    value_t prob = kappa_x / (1 + exp_bAomega[i]);
    gap += prob * log_kappa + (1 - prob) * log1p(exp_bAomega[i] - kappa_x) - labels[i] * Aomega[i];

  }
  if (gap != gap) {
    throw gap;
  }

  gap += l1_penalty * l1_norm(omega);

  obj_vals.set_primal_obj_x(obj_vals.dual_obj() + gap);
}


void SparseLogRegSolver::update_newton_2nd_derivatives(value_t epsilon_to_add) {
  for (index_t i = 0; i < num_examples; ++i) {
    value_t label = data->b_value(i);
    value_t prob = 1 / (1 + exp_bAomega[i]);
    newton_2nd_derivatives[i] = label * label * prob * (1 - prob) + epsilon_to_add;
  }
  sum_newton_2nd_derivatives = sum_vector(newton_2nd_derivatives);
}

}
