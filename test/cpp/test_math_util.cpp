#include "catch.hpp"

#include <blitzml/base/common.h>
#include <blitzml/base/math_util.h>
#include <vector>

using std::vector;

using namespace BlitzML;

TEST_CASE( "test_sq", "[math_util]" ) {
  REQUIRE( sq(0) == 0 );
  REQUIRE( sq(2) == 4 );
  value_t v = -2.5;
  REQUIRE( sq(v) == 6.25 );
}

TEST_CASE( "test_cube", "[math_util]" ) {
  REQUIRE( cube(0) == 0 );
  REQUIRE( cube(2) == 8 );
  value_t v = -2.5;
  REQUIRE( cube(v) == -15.625 );
}

TEST_CASE( "test_sign", "[math_util]" ) {
  REQUIRE( sign(-0.5) == -1 );
  REQUIRE( sign(-100) == -1 );
  REQUIRE( sign(0.01) == 1 );
  REQUIRE( sign(2) == 1 );
}

TEST_CASE( "l2_norm_sq", "[math_util]" ) {
  value_t values[5] = {1.0, -0.5, 0., -2.0, 3.0};
  REQUIRE( l2_norm_sq(values, 5) == 14.25 );
  value_t values2[1] = {-2.5};
  REQUIRE( l2_norm_sq(values2, 1) == 6.25 );

  vector<value_t> values_v(values, values + 5);
  REQUIRE( l2_norm_sq(values_v) == 14.25 );

  vector<value_t> values2_v(values2, values2 + 1);
  REQUIRE( l2_norm_sq(values2_v) == 6.25 );
}

TEST_CASE( "l2_diff_norm", "[math_util]" ) {
  vector<value_t> v1;
  vector<value_t> v2;
  REQUIRE( l2_norm_diff_sq(v1, v2) == 0. );
  REQUIRE( l2_norm_diff(v1, v2) == 0. );
  v1.push_back(-0.5); v2.push_back(0.);
  REQUIRE( l2_norm_diff_sq(v1, v2) == 0.25 );
  REQUIRE( l2_norm_diff(v1, v2) == 0.5 );
  v1.push_back(-1.5); v2.push_back(-1.5);
  REQUIRE( l2_norm_diff_sq(v1, v2) == 0.25 );
  REQUIRE( l2_norm_diff(v1, v2) == 0.5 );
  v1.push_back(10.5); v2.push_back(0.5);
  REQUIRE( l2_norm_diff_sq(v1, v2) == 100.25 );
  REQUIRE( sq(l2_norm_diff(v1, v2)) == Approx( 100.25 ) );
}

TEST_CASE( "l1_norm", "[math_util]" ) {
  vector<value_t> v;
  value_t* vp;
  REQUIRE( l1_norm(v) == 0. );
  REQUIRE( l1_norm(vp, 0) == 0. );
  v.push_back(-5.0);
  REQUIRE( l1_norm(v) == 5. );
  REQUIRE( l1_norm(&v[0], 1) == 5. );
  v.push_back(0.5);
  REQUIRE( l1_norm(v) == 5.5 );
  REQUIRE( l1_norm(&v[0], 2) == 5.5 );
}

TEST_CASE( "sum", "[math_util]" ) {
  vector<value_t> v;
  value_t* vp;
  REQUIRE( sum_vector(v) == 0. );
  REQUIRE( sum_array(vp, 0) == 0. );
  v.push_back(-5.0);
  REQUIRE( sum_vector(v) == -5. );
  REQUIRE( sum_array(&v[0], 1) == -5. );
  v.push_back(0.5);
  REQUIRE( sum_vector(v) == -4.5 );
  REQUIRE( sum_array(&v[0], 2) == -4.5 );
}

TEST_CASE( "l0_norm", "[math_util]" ) {
  vector<value_t> v;
  value_t* vp;
  REQUIRE( l0_norm(v) == 0. );
  REQUIRE( l0_norm(vp, 0) == 0. );
  v.push_back(0.);
  REQUIRE( l0_norm(v) == 0 );
  REQUIRE( l0_norm(&v[0], 1) == 0 );
  v.push_back(-2.0);
  REQUIRE( l0_norm(v) == 1 );
  REQUIRE( l0_norm(&v[0], 2) == 1 );
  v.push_back(2.6);
  REQUIRE( l0_norm(v) == 2 );
  REQUIRE( l0_norm(&v[0], 3) == 2 );
}

TEST_CASE( "inner_product", "[math_util]" ) {
  vector<value_t> v1;
  vector<value_t> v2;
  value_t* vp1;
  value_t* vp2;
  REQUIRE( inner_product(v1, v2) == 0. );
  REQUIRE( inner_product(vp1, vp2, 0) == 0. );
  v1.push_back(0.); v2.push_back(10.);
  REQUIRE( inner_product(v1, v2) == 0. );
  REQUIRE( inner_product(&v1[0], &v2[0], 1) == 0. );
  v1.push_back(-1.4); v2.push_back(1.2);
  REQUIRE( inner_product(v1, v2) == -1.68 );
  REQUIRE( inner_product(&v1[0], &v2[0], 2) == -1.68 );
  v1.push_back(2.0); v2.push_back(0.5);
  REQUIRE( inner_product(v1, v2) == Approx( -0.68 ) );
  REQUIRE( inner_product(&v1[0], &v2[0], 3) == Approx( -0.68 ) );
}

TEST_CASE( "soft_threshold", "[math_util]" ) {
  value_t v = 1.0;
  value_t t = 1.5;
  REQUIRE( soft_threshold(v, t) == 0. );
  v = 1.5;
  REQUIRE( soft_threshold(v, t) == 0. );
  v = -1.5;
  REQUIRE( soft_threshold(v, t) == 0. );
  v = -2.6;
  REQUIRE( soft_threshold(v, t) == -1.1 );
  v = 3.5;
  REQUIRE( soft_threshold(v, t) == 2.0 );
}

TEST_CASE( "max_vector", "[math_util]" ) {
  vector<value_t> v;
  v.push_back(-1.5);
  REQUIRE( max_vector(v) == -1.5 );
  REQUIRE( max_abs(v) == 1.5 );
  v.push_back(-2.5);
  REQUIRE( max_vector(v) == -1.5 );
  REQUIRE( max_abs(v) == 2.5 );
  v.push_back(0.);
  REQUIRE( max_vector(v) == 0. );
  REQUIRE( max_abs(v) == 2.5 );
  v.push_back(-100.);
  REQUIRE( max_vector(v) == 0. );
  REQUIRE( max_abs(v) == 100. );
  v.push_back(200.);
  REQUIRE( max_vector(v) == 200. );
  REQUIRE( max_abs(v) == 200. );
}

TEST_CASE( "median", "[math_util]" ) {
  vector<value_t> v;
  v.push_back(-5.0); v.push_back(2.0); v.push_back(1.0);
  REQUIRE( median(v, 0, 3) == 1.0 );
  REQUIRE( median(v, 1, 2) == 2.0 );
  REQUIRE( median_last_k(v, 3) == 1.0 );
  v.clear();
  v.push_back(-2.0); v.push_back(-3.0);
  REQUIRE( median(v, 0, 2) == -2.5 );
  v.push_back(10.0); v.push_back(-100.0);
  REQUIRE( median(v, 0, 4) == -2.5 );
  REQUIRE( median_last_k(v, 3) == -3.0 );
  REQUIRE( median_last_k(v, 2) == -45.0 );
  REQUIRE( median_last_k(v, 1) == -100.0 );
}

TEST_CASE( "quadratic_roots", "[math_util]" ) {
  value_t a = 2.0; value_t b = -8.0; value_t c = -10.0;
  std::pair<value_t, value_t> roots = compute_quadratic_roots(a, b, c);
  REQUIRE( roots.first == -1.0 );
  REQUIRE( roots.second == 5.0 );

  a = 0.0; b = -2.0; c = 5.0;
  roots = compute_quadratic_roots(a, b, c);
  REQUIRE( roots.first == 2.5 );
  REQUIRE( roots.second == 2.5 );

  a = 2.0; b = -2.0; c = -2.0;
  roots = compute_quadratic_roots(a, b, c);
  REQUIRE( roots.first == Approx( -0.618033988749895 ) );
  REQUIRE( roots.second == Approx( 1.618033988749895 ) );
}


