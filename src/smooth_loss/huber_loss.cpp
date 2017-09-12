#include <blitzml/smooth_loss/huber_loss.h>
#include <blitzml/base/math_util.h>
#include <math.h>

namespace BlitzML {

value_t HuberLoss::compute_loss(value_t a_dot_omega, value_t label) const {
  value_t residual = a_dot_omega - label;
  if (residual < -1.) {
    return -residual - 0.5;
  } else if (residual > 1.) {
    return residual - 0.5;
  } else {
    return 0.5 * sq(residual);
  }
}


value_t HuberLoss::compute_conjugate(value_t dual_variable, 
                                     value_t label) const {
  return dual_variable * label + sq(dual_variable) / 2;
}


value_t HuberLoss::compute_deriative(value_t a_dot_omega, 
                                     value_t label) const {
  value_t residual = a_dot_omega - label;
  if (residual < -1.) {
    return -1.;
  } else if (residual > 1.) {
    return 1.;
  } else {
    return residual;
  }
}


value_t HuberLoss::compute_2nd_derivative(value_t a_dot_omega,
                                             value_t label) const {
  value_t residual = a_dot_omega - label;
  if (residual < -1.) {
    return MIN_SMOOTH_LOSS_2ND_DERIVATIVE;
  } else if (residual > 1.) {
    return MIN_SMOOTH_LOSS_2ND_DERIVATIVE;
  } else {
    return 1.;
  }
}


value_t HuberLoss::lipschitz_constant() const {
  return 1;
}

} // namespace BlitzML


