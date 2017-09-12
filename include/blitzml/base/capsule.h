#pragma once

#include <blitzml/base/common.h>

namespace BlitzML {

struct CapsuleParams {
  value_t radius;
  value_t left_beta;
  value_t right_beta;
  value_t xi;

  CapsuleParams(value_t radius, value_t left_beta, value_t right_beta, value_t xi)
    : radius(radius), left_beta(left_beta), right_beta(right_beta), xi(xi) { }
  CapsuleParams() : radius(0.), left_beta(0.), right_beta(0.), xi(0.) { }

  virtual ~CapsuleParams() { }
};


class CapsuleCalculator {
  public:
    CapsuleCalculator(value_t gamma, value_t Delta, value_t d_sq);

    virtual ~CapsuleCalculator() { }

    void compute_capsule_params(value_t xi, CapsuleParams &params);

  private:
    value_t gamma, Delta, d, r;

    value_t compute_left_boundary(value_t xi);

    value_t compute_right_boundary(value_t xi);

    value_t compute_radius(value_t xi);
    
    value_t compute_tau(value_t beta, value_t xi, int shift_term_multiplier);

    value_t compute_deriv_tau(value_t beta, value_t xi, int shift_term_multiplier);

    value_t compute_max_tau(value_t xi, int shift_term_multiplier);

    value_t compute_xi_term(value_t beta, value_t xi);

    value_t compute_d_term(value_t beta, value_t xi);

    CapsuleCalculator();
};

}
