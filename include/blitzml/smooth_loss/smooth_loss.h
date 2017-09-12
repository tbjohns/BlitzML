#pragma once

#include <blitzml/base/common.h>

namespace BlitzML {

const value_t MIN_SMOOTH_LOSS_2ND_DERIVATIVE = 0.;

class SmoothLoss {
  public:
    SmoothLoss() { }

    virtual ~SmoothLoss() { }

    virtual value_t compute_loss(value_t a_dot_omega, value_t label) const = 0;
    virtual value_t compute_conjugate(value_t dual_variable, 
                                      value_t label) const = 0;
    virtual value_t compute_deriative(value_t a_dot_omega, 
                                      value_t label) const = 0;
    virtual value_t compute_2nd_derivative(value_t a_dot_omega, 
                                           value_t label) const = 0;
    virtual value_t lipschitz_constant() const = 0;
};

} // namespace BlitzML
