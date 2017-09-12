#include <blitzml/sparse_linear/sparse_linear_solver.h>

#include <blitzml/base/math_util.h>
#include <blitzml/base/vector_util.h>
#include <blitzml/base/timer.h>
#include <blitzml/base/subproblem_controller.h>

#include <blitzml/smooth_loss/squared_loss.h>
#include <blitzml/smooth_loss/logistic_loss.h>
#include <blitzml/smooth_loss/squared_hinge_loss.h>
#include <blitzml/smooth_loss/smoothed_hinge_loss.h>
#include <blitzml/smooth_loss/huber_loss.h>

using std::vector;

namespace BlitzML {


value_t SparseLinearSolver::compute_dual_obj() const { 
  value_t loss = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    loss += loss_function->compute_loss(Aomega[j], data->b_value(j));
  }
  return -(loss + l1_penalty * l1_norm(omega));
}


value_t SparseLinearSolver::compute_primal_obj_x() const { 
  value_t obj = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    obj += loss_function->compute_conjugate(kappa_x * x[j], data->b_value(j));
  }
  return obj;
}


value_t SparseLinearSolver::compute_primal_obj_y() const { 
  value_t obj = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    obj += loss_function->compute_conjugate(y[j], data->b_value(j));
  }
  return obj;
}


void SparseLinearSolver::update_subproblem_obj_vals() { 
  value_t primal_obj_x = compute_primal_obj_x();
  value_t dual_obj = compute_dual_obj();
  obj_vals.set_primal_obj_x(primal_obj_x);
  obj_vals.set_dual_obj(dual_obj);
}


void SparseLinearSolver::solve_subproblem() { 
  SubproblemController controller(subproblem_params);
  
  SubproblemState initial_state;
  set_initial_subproblem_state(initial_state);

  set_initial_subproblem_z();

  for (int newton_itr = 0; newton_itr < subproblem_params.max_iterations; ++newton_itr) {
    setup_proximal_newton_problem();
    value_t cd_threshold = compute_cd_threshold();

    scale_Delta_Aomega_by_2nd_derivatives();

    int cd_itr, max_num_cd_itr, min_num_cd_itr;
    set_max_and_min_num_cd_itr(max_num_cd_itr, min_num_cd_itr);
    Timer t1;
    for (cd_itr = 1; cd_itr <= max_num_cd_itr; ++cd_itr) {
      value_t est_grad_sq = update_coordinates_in_working_set();
      if (cd_itr >= min_num_cd_itr) {
        if (controller.should_compute_duality_gap()) {
          break;
        }
        if (est_grad_sq < cd_threshold) {
          break;
        }
      }
    }

    unscale_Delta_Aomega_by_2nd_derivatives();

    perform_backtracking();
    update_bias_subproblem();

    update_kappa_x();
    update_subproblem_obj_vals();
    update_z();

    bool sufficient_dual_progress = 
           check_sufficient_dual_progress(initial_state, subproblem_params.epsilon);

    if (controller.should_terminate(obj_vals, sufficient_dual_progress)
        || (newton_itr == subproblem_params.max_iterations)) {
      return;
    }
  }
}


value_t SparseLinearSolver::update_coordinates_in_working_set() {
  value_t ret = 0.;
  for (const_index_itr ind = ws.begin_indices(); ind != ws.end_indices(); ++ind) {
    value_t subgrad = update_feature(*ind);
    ret += sq(subgrad);
  }
  ws.shuffle_indices();
  return ret;
}


void SparseLinearSolver::
set_max_and_min_num_cd_itr(int &max_num_cd_itr, int &min_num_cd_itr) const {
  if (iteration_number == 1) {
    min_num_cd_itr = 1;
    max_num_cd_itr = 1;
  } else {
    min_num_cd_itr = 3;
    max_num_cd_itr = 64;
  }
}


value_t SparseLinearSolver::compute_cd_threshold() const {
  value_t ret = 0.;
  for (const_index_itr i = ws.begin(); i != ws.end(); ++i) {
    if (omega[*i] != 0.) {
      ret += sq(fabs(ATx[*i]) - l1_penalty);
    } else {
      ret += sq(soft_threshold(ATx[*i], l1_penalty));
    }
  }
  return 0.25 * ret;
}


value_t SparseLinearSolver::norm_diff_sq_z_initial_x(const SubproblemState& initial_state) const {
  value_t norm_diff_sq = 0.;
  for (index_t j = 0; j < num_examples; ++j) {
    norm_diff_sq += sq(initial_state.x[j] - kappa_z * z[j]);
  }
  return norm_diff_sq;
}


void SparseLinearSolver::setup_subproblem_no_working_sets() {
  Solver::setup_subproblem_no_working_sets();
  subproblem_params.max_iterations = 1;
}


void SparseLinearSolver::set_initial_subproblem_z() {
  z = y;
  for (const_index_itr i = ws.begin_sorted(); i != ws.end_sorted(); ++i) {
    ATz[*i] = ATy[*i];
  }
  kappa_z = 1.0;
  obj_vals.set_primal_obj_z(obj_vals.primal_obj_y());
  z_match_y = true;
  z_match_x = false;
}


void SparseLinearSolver::setup_proximal_newton_problem() {
  Delta_omega.assign(ws.size(), 0.);
  Delta_Aomega.assign(num_examples, 0.);
  value_t hessian_extra = 1e-12;
  update_newton_2nd_derivatives(hessian_extra);
  Delta_bias = (use_bias) ? -sum_x / sum_newton_2nd_derivatives : 0.;
  inv_newton_2nd_derivative_cache.resize(ws.size());
  col_ip_newton_2nd_derivative_cache.resize(ws.size());

  bool s_d_is_const = is_vector_const(newton_2nd_derivatives, 1e-12);

  for (size_t ind = 0; ind != ws.size(); ++ind) {
    index_t i = ws.ith_member(ind);
    const Column& col = *A_cols[i];

    value_t weighted_norm_sq = 0;
    if (s_d_is_const) {
      value_t const_val = newton_2nd_derivatives[0];
      weighted_norm_sq = const_val / (inv_lipschitz_cache[i] * loss_function->lipschitz_constant());
    } else {
      weighted_norm_sq = col.weighted_norm_sq(newton_2nd_derivatives);
    }

    if (use_bias) {
      if (s_d_is_const) {
        value_t const_val = newton_2nd_derivatives[0];
        col_ip_newton_2nd_derivative_cache[ind] = col_means_cache[i] * const_val * num_examples;
      } else {
        col_ip_newton_2nd_derivative_cache[ind] = col.inner_product(newton_2nd_derivatives);
        weighted_norm_sq += col_means_cache[i] * (col_means_cache[i] * sum_newton_2nd_derivatives - 2 * col_ip_newton_2nd_derivative_cache[ind]);
      }
    }

    if (weighted_norm_sq > 0.) {
      inv_newton_2nd_derivative_cache[ind] = 1. / weighted_norm_sq;
    } else {
      inv_newton_2nd_derivative_cache[ind] = -1.;
    }
  }
}


void SparseLinearSolver::update_z() {
  bool is_best = obj_vals.primal_obj_x() <= obj_vals.primal_obj_z();
  if (is_best) {
    obj_vals.set_primal_obj_z(obj_vals.primal_obj_x());
    z = x;
    kappa_z = kappa_x;
    for (const_index_itr i = ws.begin_sorted(); i != ws.end_sorted(); ++i) {
      ATz[*i] = ATx[*i];
    }
    z_match_x = true;
    z_match_y = false;
  } else {
    z_match_x = false;
  }
}


void SparseLinearSolver::scale_Delta_Aomega_by_2nd_derivatives() {
  for (index_t i = 0; i < num_examples; ++i) {
    Delta_Aomega[i] *= newton_2nd_derivatives[i];
  }
}


void SparseLinearSolver::unscale_Delta_Aomega_by_2nd_derivatives() {
  for (index_t i = 0; i < num_examples; ++i) {
    Delta_Aomega[i] /= newton_2nd_derivatives[i];
  }
}


value_t SparseLinearSolver::update_feature(index_t working_set_ind) {
  // note: Delta_Aomega is scaled by newton 2nd derivatives
  value_t inv_L = inv_newton_2nd_derivative_cache[working_set_ind];
  if (inv_L < 0) {
    return 0.;
  }

  index_t feature_ind = ws.ith_member(working_set_ind);
  const Column& col = *A_cols[feature_ind];
  value_t grad = ATx[feature_ind] + 
         col.inner_product(Delta_Aomega) +
         col_ip_newton_2nd_derivative_cache[working_set_ind] * Delta_bias;

  value_t current_value = omega[feature_ind] + Delta_omega[working_set_ind];
  if (current_value == 0. && fabs(grad) < l1_penalty) {
    return 0.;
  }
  value_t pre_shrink = current_value - grad * inv_L;
  value_t new_value = soft_threshold(pre_shrink, l1_penalty * inv_L);
  if (new_value == current_value) {
    return 0.;
  }

  value_t delta = new_value - current_value;
  col.weighted_add_multiple(Delta_Aomega, newton_2nd_derivatives, delta);
  Delta_omega[working_set_ind] += delta;

  if (use_bias) {
    value_t grad_bias = col_ip_newton_2nd_derivative_cache[working_set_ind]  * delta;
    Delta_bias -= grad_bias * (1 / sum_newton_2nd_derivatives);
  }

  return grad + sign(current_value) * l1_penalty;
}


void SparseLinearSolver::update_bias_subproblem() {
  update_bias();
}


void SparseLinearSolver::update_bias(int max_newton_itr) {
  if (!use_bias) {
    return;
  }

  Delta_Aomega.assign(num_examples, 0.);
  Delta_bias = 0.;

  for (int newton_itr = 0; newton_itr < max_newton_itr; ++newton_itr) {
    value_t change = perform_newton_update_on_bias();
    if (fabs(change) < 1e-14) {
      return;
    }
  }
} 


value_t SparseLinearSolver::perform_newton_update_on_bias() {
  update_newton_2nd_derivatives(1e-8);
  Delta_bias = -sum_x / sum_newton_2nd_derivatives;
  value_t step_size = 1.;
  unsigned backtrack_itr = 0;
  while (backtrack_itr < 10) {
    update_x(step_size);
    if (sum_x * Delta_bias <= 0.) {
      value_t change = step_size * Delta_bias;
      bias += change;
      add_scalar_to_vector(Aomega, change);
      return change;
    }
    ++backtrack_itr;
    step_size *= 0.5;
  }
  return 0;
}


void SparseLinearSolver::perform_backtracking() {
  value_t step_size = 1.;
  unsigned backtrack_itr = 0;
  while (++backtrack_itr) {
    update_x(step_size);

    value_t derivative = compute_backtracking_step_size_derivative(step_size);
    if (derivative <= 1e-12) {
      break;
    } else if (backtrack_itr > MAX_BACKTRACK_STEPS) {
      compute_Aomega();
      update_x(0.);
      return;
    }

    step_size *= 0.5;
  }

  for (const_index_itr ind = ws.begin_indices(); ind != ws.end_indices(); ++ind) {
    omega[ws.ith_member(*ind)] += step_size * Delta_omega[*ind];
  }
  bias += step_size * Delta_bias;
  for (index_t i = 0; i < num_examples; ++i) {
    Aomega[i] += step_size * (Delta_Aomega[i] + Delta_bias);
  }
}


value_t SparseLinearSolver::
compute_backtracking_step_size_derivative(value_t step_size) const {

  value_t derivative_loss = inner_product(Delta_Aomega, x) + Delta_bias * sum_x;

  value_t derivative_regularization = 0.;
  for (const_index_itr ind = ws.begin_indices(); ind != ws.end_indices(); ++ind) {
    index_t i = ws.ith_member(*ind);
    value_t omega_i = omega[i] + step_size * Delta_omega[*ind];
    if (fabs(omega_i) < 1e-14) {
      derivative_regularization -= l1_penalty * fabs(Delta_omega[*ind]);  
    } else {
      derivative_regularization += l1_penalty * Delta_omega[*ind] * sign(omega_i);
    }
  }

  return derivative_loss + derivative_regularization;
}


void SparseLinearSolver::update_x(value_t step_size) {
  for (index_t i = 0; i < num_examples; ++i) {
    value_t diff_a_dot_omega = step_size * (Delta_Aomega[i] + Delta_bias);
    value_t label = data->b_value(i);
    x[i] = loss_function->compute_deriative(Aomega[i] + diff_a_dot_omega, label);
  }
  sum_x = sum_vector(x);
  z_match_x = false;
  z_match_y = false;
}


void SparseLinearSolver::update_newton_2nd_derivatives(value_t epsilon_to_add) {
  for (index_t i = 0; i < num_examples; ++i) {
    value_t a_dot_omega = Aomega[i];
    value_t label = data->b_value(i);
    newton_2nd_derivatives[i] = 
          loss_function->compute_2nd_derivative(a_dot_omega, label) + epsilon_to_add;
  }
  sum_newton_2nd_derivatives = sum_vector(newton_2nd_derivatives);
}


void SparseLinearSolver::update_kappa_x() {
  value_t max_abs_grad_i = l1_penalty;
  for (const_index_itr i = ws.begin_sorted(); i != ws.end_sorted(); ++i) {
    value_t grad_i = A_cols[*i]->inner_product(x);
    ATx[*i] = grad_i;
    if (fabs(grad_i) > max_abs_grad_i) {
      max_abs_grad_i = fabs(grad_i);
    }
  }
  kappa_x = l1_penalty / max_abs_grad_i;
  if (kappa_x >= 1.) {
    kappa_x = 1.;
  }
}


value_t SparseLinearSolver::compute_alpha() { 
  update_non_working_set_gradients();

  value_t alpha = 1.;
  for (index_t j = 0; j < num_components; ++j) {
    value_t alpha_j = compute_alpha_for_feature(j);
    if (alpha_j < alpha) {
      alpha = alpha_j;
    }
  }

  if (alpha == 1) {
    ++consecutive_alpha_eq_1;
  } else {
    consecutive_alpha_eq_1 = 0;
  }

  return alpha;
}


void SparseLinearSolver::update_non_working_set_gradients() {
  // x update:
  for (index_t i = 0; i < num_components; ++i) {
    if (!ws.is_in_working_set(i)) {
      ATx[i] = A_cols[i]->inner_product(x);
    }
  }

  // z update:
  if (z_match_y) {
    ATz = ATy;
  } else if (z_match_x) {
    ATz = ATx;
  } else {
    for (index_t i = 0; i < num_components; ++i) {
      if (!ws.is_in_working_set(i)) {
        ATz[i] = A_cols[i]->inner_product(z);
      }
    }
  }
}


value_t SparseLinearSolver::compute_alpha_for_feature(index_t i) const {
  if (ws.is_in_working_set(i)) {
    return 1.;
  }
  value_t kappa_x_AiTz = kappa_z * ATz[i];
  if (fabs(kappa_x_AiTz) <= l1_penalty) {
    return 1.;
  }
  value_t AiTy = ATy[i];
  if (kappa_x_AiTz == AiTy) {
    return 1.;
  }
  value_t value = (kappa_x_AiTz < 0) ? -l1_penalty : l1_penalty;
  return (value - AiTy) / (kappa_x_AiTz - AiTy);
}


void SparseLinearSolver::update_y() { 
  value_t alpha_kappa_z = alpha * kappa_z;
  for (index_t j = 0; j < num_examples; ++j) {
    y[j] = alpha_kappa_z * z[j] + (1 - alpha) * y[j];
  }
  for (index_t i = 0; i < num_components; ++i) {
    ATy[i] = alpha_kappa_z * ATz[i] + (1 - alpha) * ATy[i];
  }
}


unsigned short SparseLinearSolver::compute_priority_value(index_t i) const {
  if (omega[i] != 0.) {
    return 0;
  }

  value_t ATyi = ATy[i];
  value_t ATxi = ATx[i];
  value_t diffi = ATxi - ATyi;
  value_t inv_norm_sq = inv_lipschitz_cache[i] * loss_function->lipschitz_constant();

  unsigned short top = capsule_candidates.size();
  unsigned short bottom = 0;
  bool in_working_set = false;
  do {
    unsigned short ind = (top + bottom) / 2;
    const CapsuleParams& cap = capsule_candidates[ind];

    value_t beta_l = cap.left_beta;
    value_t val_l = ATyi + beta_l * diffi;

    value_t beta_r = cap.right_beta;
    value_t val_r = ATyi + beta_r * diffi;

    value_t val = std::max(std::fabs(val_l), std::fabs(val_r));
    value_t dist_sq = (val >= l1_penalty) ? 0.
                    : sq(l1_penalty - val) * inv_norm_sq;

    in_working_set = (dist_sq <= sq(cap.radius));
    if (in_working_set) {
      top = ind;
    } else {
      bottom = ind;
    }
  } while (top > bottom + 1);

  return top;
}


void SparseLinearSolver::screen_components() { 
  vector<bool> should_screen;
  bool any_screened = mark_components_to_screen(should_screen);
  if (any_screened) {
    apply_screening_result(should_screen);
  }
}


bool SparseLinearSolver::mark_components_to_screen(vector<bool> &should_screen) const {
  value_t d_sq = compute_d_sq();
  value_t thresh_sq = obj_vals.duality_gap() - d_sq / 4;
  if (thresh_sq <= 0.) {
    return false;
  }

  if (sq(l1_penalty) * max_inv_lipschitz_cache < thresh_sq) {
    return false;
  }

  should_screen.assign(num_components, false);
  bool any_screened = false;
  for (index_t i = 0; i < num_components; ++i) {
    if (inv_lipschitz_cache[i] < 0) {
      should_screen[i] = true;
      any_screened = true;
    }

    if (omega[i] != 0.) {
      continue;
    }

    value_t val = (ATx[i] + ATy[i]) / 2; 
    value_t dist = l1_penalty - fabs(val);
    if (dist < 0.) {
      continue;
    }

    bool screen_i = (sq(dist) * inv_lipschitz_cache[i] >= thresh_sq);
    if (screen_i) {
      should_screen[i] = true;
      any_screened = true;
    }
  }

  return any_screened;
}


void SparseLinearSolver::apply_screening_result(const vector<bool> &should_screen) {
  index_t i = 0;
  for (index_t ind = 0; ind < num_components; ++ind) {
    if (!should_screen[ind]) {
      ATy[i] = ATy[ind];
      ATx[i] = ATx[ind];
      omega[i] = omega[ind];
      inv_lipschitz_cache[i] = inv_lipschitz_cache[ind];
      col_means_cache[i] = col_means_cache[ind];
      A_cols[i] = A_cols[ind];
      screen_indices_map[i] = screen_indices_map[ind];
      ++i;
    }
  }
  num_components = i;
  omega.resize(num_components);
  ws.reduce_max_size(num_components);
  if (!use_working_sets()) {
    ws.clear();
    for (index_t i = 0; i < num_components; ++i) {
      ws.add_index(i);
    }
  }
}


size_t SparseLinearSolver::size_of_component(index_t i) const {
  return A_cols[i]->nnz();
}


index_t SparseLinearSolver::initial_problem_size() const { 
  return data->num_cols();
}


void SparseLinearSolver::initialize(value_t *initial_conditions) { 
  deserialize_parameters();
  set_data_dimensions();
  initialize_proximal_newton_variables();
  cache_feature_info();
  initialize_blitz_variables(initial_conditions);
}


value_t SparseLinearSolver::strong_convexity_constant() const {
  return 1 / loss_function->lipschitz_constant();
}


void SparseLinearSolver::deserialize_parameters() {
  l1_penalty = (*params)[0];
  if ((*params)[2] != 0.) {
    use_bias = true;
  } else {
    use_bias = false;
  }
  int loss_type = static_cast<int>((*params)[3]);
  delete_loss_function();
  switch (loss_type) {
    case 0:
      loss_function = new SquaredLoss();
      break;
    case 1:
      loss_function = new HuberLoss();
      break;
    case 2:
      loss_function = new LogisticLoss();
      break;
    case 3:
      loss_function = new SquaredHingeLoss();
      break;
    case 4:
      loss_function = new SmoothedHingeLoss();
      break;
    default:
      loss_function = new SquaredLoss();
      break;
  }
}


void SparseLinearSolver::set_data_dimensions() {
  num_examples = data->num_rows();
  num_components = data->num_cols();
}


void SparseLinearSolver::initialize_blitz_variables(
    value_t* initial_conditions) {
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

  update_x();
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


void SparseLinearSolver::compute_Aomega() {
  Aomega.assign(num_examples, bias);
  for (index_t i = 0; i < num_components; ++i) {
    if (omega[i] != 0.) {
      A_cols[i]->add_multiple(Aomega, omega[i]);
    }
  }
}


void SparseLinearSolver::cache_feature_info() {
  A_cols.resize(num_components);
  inv_lipschitz_cache.resize(num_components);
  col_means_cache.resize(num_components);
  value_t lipschitz_constant = loss_function->lipschitz_constant();
  for (index_t i = 0; i < num_components; ++i) {
    A_cols[i] = data->column(i);
    const Column& col = *A_cols[i];
    value_t norm_sq = col.l2_norm_sq();
    value_t mean = col.mean();
    if (use_bias) {
      norm_sq -= num_examples * sq(mean);
    }
    if (norm_sq <= 1e-12) {
      norm_sq = -1.;
    }
    inv_lipschitz_cache[i] = 1 / (norm_sq * lipschitz_constant);
    col_means_cache[i] = mean;
  }

  max_inv_lipschitz_cache = max_vector(inv_lipschitz_cache);
}


void SparseLinearSolver::initialize_proximal_newton_variables() {
  newton_2nd_derivatives.resize(num_examples);
  Delta_Aomega.assign(num_examples, 0);
  Delta_bias = 0.;
}


void SparseLinearSolver::log_variables(Logger &logger) const { 
  std::vector<index_t> indices;
  std::vector<value_t> values;
  size_t nnz_weights = l0_norm(omega);
  indices.reserve(nnz_weights);
  values.reserve(nnz_weights);
  for (index_t k = 0; k < num_components; ++k) {
    if (omega[k] != 0.) {
      indices.push_back(screen_indices_map[k]);
      values.push_back(omega[k]);
    }
  }
  if (log_vectors()) {
    logger.log_vector<index_t>("weight_indices", indices);
    logger.log_vector<value_t>("weight_values", values);
    logger.log_vector<value_t>("z", z);
  }
  logger.log_value<value_t>("bias", bias);
  logger.log_value<value_t>("l1_penalty", l1_penalty);
  logger.log_value<size_t>("number_nonzero_weights", nnz_weights);
}


void SparseLinearSolver::fill_result(value_t *result) const { 
  size_t ind = 0;

  // fill in weights
  for (index_t i = 0; i < data->num_cols(); ++i) {
    result[ind++] = 0.;
  }
  for (size_t ind = 0; ind < omega.size(); ++ind) {
    result[screen_indices_map[ind]] = omega[ind];
  }
  result[ind++] = bias;

  // fill in dual solution
  for (index_t i = 0; i < num_examples; ++i) {
    result[ind++] = y[i];
  }

  result[ind++] = obj_vals.duality_gap();
  result[ind++] = obj_vals.dual_obj();
}


value_t SparseLinearSolver::compute_max_l1_penalty(
    const Dataset *data, const Parameters *params) {
  this->data = data;
  this->params = params;

  initialize(NULL);

  for (index_t j = 0; j < num_components; ++j) {
    ATx[j] = A_cols[j]->inner_product(x);
  }
  return max_abs(ATx);
}


void SparseLinearSolver::delete_loss_function() {
  if (loss_function != NULL) {
    delete loss_function;
    loss_function = NULL;
  }
}


} // namespace BlitzML

