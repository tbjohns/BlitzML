#include <blitzml/smooth_loss/squared_hinge_loss.h>
#include <blitzml/base/math_util.h>
#include <math.h>

namespace BlitzML {

value_t SquaredHingeLoss::compute_loss(value_t a_dot_omega, value_t label) const {
  value_t one_minus_label_times_aomega = 1 - label * a_dot_omega;
  if (one_minus_label_times_aomega <= 0.)
    return 0;
  return 0.5 * sq(one_minus_label_times_aomega);
}


value_t SquaredHingeLoss::compute_conjugate(value_t dual_variable, 
                                        value_t label) const {
  value_t one_plus_ratio = 1 + dual_variable / label;
  return (sq(one_plus_ratio) - 1) / 2;
}


value_t SquaredHingeLoss::compute_deriative(value_t a_dot_omega, 
                                        value_t label) const {
  value_t one_minus_label_times_aomega = 1 - label * a_dot_omega;
  if (one_minus_label_times_aomega <= 0.)
    return 0.;
  return -label * one_minus_label_times_aomega;
}


value_t SquaredHingeLoss::compute_2nd_derivative(value_t a_dot_omega,
                                             value_t label) const {
  value_t one_minus_label_times_aomega = 1 - label * a_dot_omega;
  if (one_minus_label_times_aomega <= 0.) {
    return MIN_SMOOTH_LOSS_2ND_DERIVATIVE;
  } else {
    return sq(label);
  }
}


value_t SquaredHingeLoss::lipschitz_constant() const {
  return 1;
}

} // namespace BlitzML

