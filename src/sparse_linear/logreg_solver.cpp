#include <blitzml/sparse_linear/logreg_solver.h>

#include <blitzml/base/math_util.h>
#include <blitzml/base/vector_util.h>

#include "math.h"

namespace BlitzML {

void SparseLogRegSolver::initialize_blitz_variables(value_t* initial_conditions) {
  if (initial_conditions != NULL) {
    omega.assign(initial_conditions, initial_conditions + num_components);
  } else {
    omega.assign(num_components, 0.);
  }
  bias = 0.;
  compute_Aomega();

  x.resize(num_examples);
  ATx.resize(num_components);
  kappa_x = 1.;

  y.assign(num_examples, 0.);
  ATy.assign(num_components, 0.);

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
  update_bias(10);

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

void SparseLogRegSolver::update_bias(int max_newton_itr) {
  if (!use_bias) {
    return;
  }
  value_t change = 0;
  value_t deriv = compute_deriv_bias(change);
  value_t hess = compute_hess_bias(change) + 1e-12;
  for (int newton_itr = 0; newton_itr < max_newton_itr; ++newton_itr) {
    value_t delta = -deriv / hess;

    int backtrack_itr = 0;
    while (++backtrack_itr) {
      value_t new_deriv = compute_deriv_bias(change + delta);
      if (-deriv * new_deriv < 0.5 * sq(deriv)) {
        deriv = new_deriv;
        break;
      }
      delta *= 0.5;
      if (backtrack_itr == 3) {
        delta = 0.;
        break;
      }
    } 
    change += delta;

    if (delta == 0.) {
      break;
    }
  }

  bias += change;
  
  value_t exp_change = exp(change);
  value_t inv_exp_change = 1 / exp_change;
  const value_t* labels = data->b_values();
  int count = 0;
  for (index_t i = 0; i < num_examples; ++i) {
    Aomega[i] += change;
    value_t label = labels[i];
    if (label == -1) {
      exp_bAomega[i] *= inv_exp_change;
    } else if (label == 1) {
      exp_bAomega[i] *= exp_change;
    } else {
      exp_bAomega[i] = exp(label * Aomega[i]);
      ++count;
    }

    x[i] = -label * (1 / (1 + exp_bAomega[i]));
  }
  sum_x = sum_vector(x);
  z_match_x = false;
}

void SparseLogRegSolver::perform_backtracking() {
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
      } else if (backtrack_itr >= 20) {
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


value_t SparseLogRegSolver::compute_deriv_bias(value_t change) const {
  value_t deriv = 0.;
  value_t exp_change = exp(change);
  value_t inv_exp_change = 1 / exp_change;
  const value_t* b_values = data->b_values();

  for (index_t i = 0; i < num_examples; ++i) {
    value_t label = b_values[i];
    value_t exp_bAomega_i = exp_bAomega[i];
    if (label == -1) {
      exp_bAomega_i *= inv_exp_change;
    } else if (label == 1) {
      exp_bAomega_i *= exp_change;
    } else {
      exp_bAomega_i *= exp(label * change);
    }

    value_t prob = 1 / (1 + exp_bAomega_i);
    deriv -= label * prob;
  }
  return deriv;
}


value_t SparseLogRegSolver::compute_hess_bias(value_t change) const {

  value_t hess = 0.;

  value_t exp_change = exp(change);
  value_t inv_exp_change = 1 / exp_change;
  const value_t* b_values = data->b_values();
  for (index_t i = 0; i < num_examples; ++i) {
    value_t label = b_values[i];
    value_t exp_bAomega_i = exp_bAomega[i];
    if (label == -1) {
      exp_bAomega_i *= inv_exp_change;
    } else if (label == 1) {
      exp_bAomega_i *= exp_change;
    } else {
      exp_bAomega_i *= exp(label * change);
    }
    value_t prob = 1 / (1 + exp_bAomega_i);
    hess += sq(label) * prob * (1 - prob);
  }
  return hess;
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
