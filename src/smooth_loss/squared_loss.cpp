#include <blitzml/smooth_loss/squared_loss.h>
#include <blitzml/base/math_util.h>

namespace BlitzML {

value_t SquaredLoss::compute_loss(value_t a_dot_omega, value_t label) const {
  value_t residual = a_dot_omega - label;
  return sq(residual) / 2;
}


value_t SquaredLoss::compute_conjugate(value_t dual_variable, 
                                       value_t label) const { 
  return dual_variable * label + sq(dual_variable) / 2;
}


value_t SquaredLoss::compute_deriative(value_t a_dot_omega, 
                                       value_t label) const {
  return a_dot_omega - label;
}


value_t SquaredLoss::compute_2nd_derivative(value_t a_dot_omega, 
                                            value_t label) const {
  return 1.;
}


value_t SquaredLoss::compute_bias_update(
    const std::vector<value_t> &dual_values) const {
  value_t delta = -sum_vector(dual_values) / dual_values.size();
  return delta;
}


value_t SquaredLoss::lipschitz_constant() const {
  return 1.;
}

} // namespace BlitzML

