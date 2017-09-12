#include <blitzml/sparse_linear/lasso_solver.h>

#include <blitzml/base/math_util.h>
#include <blitzml/base/vector_util.h>
#include <blitzml/base/subproblem_controller.h>

using std::vector;

namespace BlitzML {

value_t LassoSolver::compute_dual_obj() const { 
  value_t loss = 0.5 * l2_norm_sq(x);
  return -(loss + l1_penalty * l1_norm(omega));
}


value_t LassoSolver::compute_primal_obj_x() const {
  value_t ip = inner_product(data->b_values(), &x[0], num_examples);
  return 0.5 * sq(kappa_x) * l2_norm_sq(x) + kappa_x * ip;
}


value_t LassoSolver::compute_primal_obj_y() const {
  value_t ip = inner_product(data->b_values(), &y[0], num_examples);
  return 0.5 * l2_norm_sq(y) + ip;
}


value_t LassoSolver::update_coordinates_in_working_set() {
  value_t ret = 0.;
  for (const_index_itr i = ws.begin(); i != ws.end(); ++i) {
    value_t subgrad = update_feature_lasso(*i);
    ret += sq(subgrad);
  }
  ws.shuffle();
  return ret;
}


inline value_t LassoSolver::update_feature_lasso(index_t i) {
  value_t inv_L = inv_lipschitz_cache[i];
  if (inv_L < 0) {
    return 0.;
  }

  const Column& col = *A_cols[i];
  value_t current_value = omega[i];
  value_t grad = col.inner_product(x) 
                          + num_examples * Delta_bias * col_means_cache[i];
  if (current_value == 0. && fabs(grad) < l1_penalty) {
    return 0.;
  }

  value_t pre_shrink = current_value - grad * inv_L;
  value_t new_value = soft_threshold(pre_shrink, l1_penalty * inv_L);
  value_t delta = new_value - current_value;
  if (delta == 0.) {
    return 0.;
  }
  col.add_multiple(x, delta);
  omega[i] = new_value;
  if (use_bias) {
    Delta_bias -= col_means_cache[i] * delta;
  }

  return grad + sign(current_value) * l1_penalty;
}


void LassoSolver::update_bias(int max_newton_itr) {
  if (!use_bias) {
    return;
  }
  value_t grad = sum_vector(x);
  value_t delta = -grad / x.size();
  bias += delta;
  add_scalar_to_vector(x, delta);
} 


void LassoSolver::perform_backtracking() {
  if (use_bias) {
    add_scalar_to_vector(x, Delta_bias);
    bias += Delta_bias;
  }
}


void LassoSolver::setup_proximal_newton_problem() { 
  Delta_bias = 0.;
}

}

