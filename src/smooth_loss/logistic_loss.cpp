#include <blitzml/smooth_loss/logistic_loss.h>
#include <blitzml/base/math_util.h>
#include <math.h>

namespace BlitzML {

value_t LogisticLoss::compute_loss(value_t a_dot_omega, value_t label) const {
  return log1p(exp(-label * a_dot_omega));
}


value_t LogisticLoss::compute_conjugate(value_t dual_variable, 
                                        value_t label) const {
  value_t neg_ratio = -dual_variable / label;
  return neg_ratio * log(neg_ratio) + (1 - neg_ratio) * log(1 - neg_ratio);
}


value_t LogisticLoss::compute_deriative(value_t a_dot_omega, 
                                        value_t label) const {
  value_t exp_value = exp(label * a_dot_omega);
  value_t prob = 1 / (1 + exp_value);
  return -label * prob;
}


value_t LogisticLoss::compute_2nd_derivative(value_t a_dot_omega,
                                             value_t label) const {
  value_t exp_value = exp(label * a_dot_omega);
  value_t prob = 1 / (1 + exp_value);
  return label * label * prob * (1 - prob);
}


value_t LogisticLoss::lipschitz_constant() const {
  return 0.25;
}

} // namespace BlitzML
