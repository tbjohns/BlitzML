#include <blitzml/base/capsule.h>

#include <blitzml/base/math_util.h>

namespace BlitzML {

CapsuleCalculator::CapsuleCalculator(value_t gamma, value_t Delta, value_t d_sq) 
                : gamma(gamma), Delta(Delta) { 
  warn_if(Delta <= -1e-7, "capsule calculations assume Delta > 0");
  warn_if(d_sq < 0, "capsule calculations assume d_sq >= 0");

  r = gamma * d_sq / Delta / 2;
  d = sqrt(d_sq);
}

void CapsuleCalculator::compute_capsule_params(value_t xi, CapsuleParams &params) {
  params.radius = compute_radius(xi);
  params.xi = xi;
  if (d == 0.) {
    params.left_beta = 0;
    params.right_beta = 0;
  } else {
    value_t left = compute_left_boundary(xi);
    left = (left > 0.) ? 0. : left;
    params.left_beta = (left + params.radius) / d;
    value_t right = compute_right_boundary(xi);
    params.right_beta = (right - params.radius) / d;
  }
}

value_t CapsuleCalculator::compute_radius(value_t xi) {
  return compute_max_tau(xi, 0);
}

value_t CapsuleCalculator::compute_left_boundary(value_t xi) {
  return -compute_max_tau(xi, -1);
}

value_t CapsuleCalculator::compute_right_boundary(value_t xi) {
  return compute_max_tau(xi, 1);
}

value_t CapsuleCalculator::compute_tau(value_t beta, value_t xi, int shift_term_multiplier) {
  value_t xi_term = compute_xi_term(beta, xi);
  value_t d_term = compute_d_term(beta, xi);
  value_t sqrt_term = 1 + d_term - xi_term;
  value_t pre_sqrt = 2 * Delta * sqrt_term / gamma;
  value_t tau = 0.;
  if (pre_sqrt > 0) {
    tau = beta * sqrt(pre_sqrt);
  }
  return tau + beta * shift_term_multiplier * d;
}

value_t CapsuleCalculator::compute_deriv_tau(value_t beta, value_t xi, 
                              int shift_term_multiplier) {
  value_t min_deriv = 1e-6;
  if (beta == 0.)
    return min_deriv;
  if (beta == 0.5) {
    if (xi < 1.0) {
      return -min_deriv;
    } else {
      return min_deriv;
    }
  }

  value_t xi_term = compute_xi_term(beta, xi);
  value_t d_term = compute_d_term(beta, xi);
  value_t sqrt_term = 1 + d_term - xi_term;
  if (sqrt_term < 0) {
    return -min_deriv;
  }

  value_t right_term = 1 + beta * ((1 - r) / sq(1 - beta) - 2 * (1 - xi) / sq(1 - 2 * beta)) / 2 / sqrt_term;
  value_t exclude_shift = sqrt(2 * Delta * sqrt_term / gamma) * right_term;

  return exclude_shift + shift_term_multiplier * d;
}

value_t CapsuleCalculator::compute_max_tau(value_t xi, int shift_term_multiplier) {
  value_t beta_below = 0.;
  value_t beta_above = 0.5;
  for (int itr = 0; itr < 25; ++itr) {
    value_t beta_middle = (beta_below + beta_above) / 2;
    value_t deriv = compute_deriv_tau(beta_middle, xi, shift_term_multiplier);
    if (deriv < 0) {
      beta_above = beta_middle;
    } else {
      beta_below = beta_middle;
    }
  }

  // Return upper bound on solution using log convexity:
  value_t value_below = compute_tau(beta_below, xi, shift_term_multiplier);
  if (value_below <= 0.) {
    return 0;
  }
  value_t deriv_below = compute_deriv_tau(beta_below, xi, shift_term_multiplier);
  value_t theta_above = beta_above / (1 - beta_above);
  value_t theta_below = beta_below / (1 - beta_below);
  value_t deriv_log_below_theta = deriv_below * sq(1 - beta_below) / value_below; 
  value_t tau_upper_est = value_below * exp(deriv_log_below_theta * (theta_above - theta_below) / 2);
  return tau_upper_est;
}


value_t CapsuleCalculator::compute_xi_term(value_t beta, value_t xi) {
  if (xi == 1.0) {
    return 0;
  }
  return (1 - xi) / (1 - 2 * beta);
}

value_t CapsuleCalculator::compute_d_term(value_t beta, value_t xi) {
  return beta / (1 - beta) * (1 - r);
}

}

