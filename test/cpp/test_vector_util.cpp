#include "catch.hpp"

#include <blitzml/base/common.h>
#include <blitzml/base/vector_util.h>
#include <vector>
#include <algorithm>

using std::vector;

using namespace BlitzML;

TEST_CASE( "test_crude_shuffle", "[vector_util]" ) {
  vector<int> v(100, 0);
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = i;
  }
  crude_shuffle(v, 0, v.size());
  bool is_sorted = true;
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] < v[i -1]) {
      is_sorted = false;
      break;
    }
  }
  REQUIRE( is_sorted == false );
}

TEST_CASE( "indirect_sort_indices", "[vector_util]" ) {
  vector<index_t> indices;
  indices.push_back(1);
  indices.push_back(0);
  indices.push_back(3);
  indices.push_back(2);
  vector<value_t> values;
  values.push_back(5.0);
  values.push_back(1.1);
  values.push_back(1.3);
  values.push_back(1.4);
  indirect_sort_indices(indices, values);
  REQUIRE( indices[0] == 1 );
  REQUIRE( indices[1] == 2 );
  REQUIRE( indices[2] == 3 );
  REQUIRE( indices[3] == 0 );
}

TEST_CASE( "scale_vector", "[vector_util]" ) {
  vector<double> v;
  v.push_back(2.0);
  v.push_back(-1.0);
  scale_vector(v, -2.0);
  REQUIRE( v[0] == Approx(-4.0) );
  REQUIRE( v[1] == Approx(2.0) );
  v.push_back(3.5);
  scale_vector(v, 0.5);
  REQUIRE( v[0] == Approx(-2.0) );
  REQUIRE( v[1] == Approx(1.0) );
  REQUIRE( v[2] == Approx(1.75) );

  add_scalar_to_vector(v, -0.5);
  REQUIRE( v[0] == Approx(-2.5) );
  REQUIRE( v[1] == Approx(0.5) );
  REQUIRE( v[2] == Approx(1.25) );
}

TEST_CASE( "is_vector_const", "[vector_util]" ) {
  vector<double> v(100, -25.0);
  REQUIRE( is_vector_const(v) == true );
  v[0] = 25.;
  REQUIRE( is_vector_const(v) == false );
  v[0] = -25.; 
  REQUIRE( is_vector_const(v) == true );
  v[99] = 25.;
  REQUIRE( is_vector_const(v) == false );
}

