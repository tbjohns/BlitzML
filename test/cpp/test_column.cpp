#include "catch.hpp"

#include <blitzml/base/common.h>
#include <blitzml/dataset/sparse_dataset.h>
#include <blitzml/dataset/dense_dataset.h>
#include <blitzml/dataset/sparse_binary_dataset.h>

using std::vector;

using namespace BlitzML;

TEST_CASE( "test_sparse_column", "[column]" ) {
  index_t indices[5] = {1, 3, 5, 7, 9};
  float data[5] = {-1.0, 2.6, 8.1, -0.1, 0.2};
  SparseColumn<float> col(indices, data, 5, 12);

  vector<value_t> v(12, 2.0);
  v[1] = 0.1; v[3] = -5.0; v[5] = 2.0; v[7] = 0.8; v[9] = -0.1;
  REQUIRE( col.inner_product(v) == Approx(3.) );
  
  vector<value_t> w(12, 7.0);
  w[1] = 0.6; w[3] = 0.4; w[5] = -0.1; w[7] = 10.; w[9] = 0.5;
  REQUIRE( col.weighted_inner_product(v, w) == Approx(-7.69) );
  REQUIRE( col.weighted_norm_sq(w) == Approx(-3.137) );

  col.add_multiple(v, 2.8);
  REQUIRE( v[0] == 2.0 );
  REQUIRE( v[1] == Approx(-2.7) );
  REQUIRE( v[7] == Approx(0.52) );
  REQUIRE( v[9] == Approx(0.46) );
  REQUIRE( v[11] == 2.0 );
  col.add_multiple(v, -2.8);

  col.weighted_add_multiple(v, w, -2.8);
  REQUIRE( v[0] == 2.0 );
  REQUIRE( v[1] == Approx(1.78) );
  REQUIRE( v[7] == Approx(3.6) );
  REQUIRE( v[9] == Approx(-0.38) );
  REQUIRE( v[11] == 2.0 );

  REQUIRE( col.sum() == Approx(9.8) );
  REQUIRE( col.mean() == Approx(9.8 / 12.) );
  REQUIRE( col.l2_norm_sq() == Approx(73.42) );

  REQUIRE( col.indices_begin()[0] == 1 );
  REQUIRE( col.indices_begin()[4] == 9 );
}

TEST_CASE( "empty_sparse_column", "[column]" ) {
  index_t* indices;
  int* data;
  SparseColumn<int> col(indices, data, 0, 100);

  vector<value_t> v(100, 2.0);
  vector<value_t> w(100, 4.0);
  REQUIRE( col.inner_product(v) == 0. );
  REQUIRE( col.weighted_inner_product(v, w) == 0. );
  REQUIRE( col.weighted_norm_sq(w) == 0. );
  REQUIRE( col.sum() == 0. );
  REQUIRE( col.mean() == 0. );
  REQUIRE( col.l2_norm_sq() == 0. );
  REQUIRE( col.indices_begin() == col.indices_end() );
}

TEST_CASE( "sparse_column_types", "[column]" ) {
  index_t indices[2] = {0, 2};
  double data_double[2] = {1.0, 3.0}; 
  float data_float[2] = {1.0, 3.0}; 
  int data_int[2] = {1.0, 3.0}; 
  bool data_bool[2] = {true, false}; 

  SparseColumn<double> col_double(indices, data_double, 2, 3);
  SparseColumn<float> col_float(indices, data_float, 2, 3);
  SparseColumn<int> col_int(indices, data_int, 2, 3);
  SparseColumn<bool> col_bool(indices, data_bool, 2, 3);

  REQUIRE( col_int.mean() == 4./3 );
  REQUIRE( col_int.mean() == 4./3 );
  REQUIRE( col_int.mean() == 4./3 );
  REQUIRE( col_bool.mean() == 1./3 );
}

TEST_CASE( "test_binary_column", "[column]" ) {
  index_t indices[5] = {1, 3, 5, 7, 9};
  SparseBinaryColumn col(indices, 5, 12);

  vector<value_t> v(12, 2.0);
  v[1] = 0.1; v[3] = -5.0; v[5] = 2.0; v[7] = 0.8; v[9] = -0.1;
  REQUIRE( col.inner_product(v) == Approx(-2.2) );
  
  vector<value_t> w(12, 7.0);
  w[1] = 0.6; w[3] = 0.4; w[5] = -0.1; w[7] = 10.; w[9] = 0.5;
  REQUIRE( col.weighted_inner_product(v, w) == Approx(5.81) );
  REQUIRE( col.weighted_norm_sq(w) == Approx(11.4) );

  col.add_multiple(v, 2.8);
  REQUIRE( v[0] == 2.0 );
  REQUIRE( v[1] == Approx(2.9) );
  REQUIRE( v[7] == Approx(3.6) );
  REQUIRE( v[9] == Approx(2.7) );
  REQUIRE( v[11] == 2.0 );
  col.add_multiple(v, -2.8);

  col.weighted_add_multiple(v, w, -2.8);
  REQUIRE( v[0] == 2.0 );
  REQUIRE( v[1] == Approx(-1.58) );
  REQUIRE( v[7] == Approx(-27.2) );
  REQUIRE( v[9] == Approx(-1.5) );
  REQUIRE( v[11] == 2.0 );

  REQUIRE( col.sum() == Approx(5) );
  REQUIRE( col.mean() == Approx(5. / 12.) );
  REQUIRE( col.l2_norm_sq() == Approx(5.) );

  REQUIRE( col.indices_begin()[0] == 1 );
  REQUIRE( col.indices_begin()[4] == 9 );

  index_t* empty_indices;
  SparseBinaryColumn col_empty(empty_indices, 0, 10);
  REQUIRE( col_empty.weighted_inner_product(v, w) == 0. );
  REQUIRE( col_empty.mean() == 0. );
  REQUIRE( col_empty.indices_begin() == col_empty.indices_end() );
}

TEST_CASE( "dense_column", "[column]" ) {
  float data[4] = {0.5, -0.1, 2.9, -8.1};
  DenseColumn<float> col(data, 4);

  vector<value_t> v(4, 0.);
  v[0] = -1.0; v[1] = 3.0; v[2] = 9.9; v[3] = -0.1;
  vector<value_t> w(4, 0.);
  w[0] = 2.2; w[1] = -0.2; w[2] = 0.0; w[3] = -0.9;

  REQUIRE( col.inner_product(v) == Approx(28.72) );
  REQUIRE( col.weighted_inner_product(v, w) == Approx(-1.769) );
  REQUIRE( col.weighted_norm_sq(w) == Approx( -58.501 ) );

  col.add_multiple(v, 5.5);
  REQUIRE( v[0] == Approx(1.75) );
  REQUIRE( v[2] == Approx(25.85) );
  REQUIRE( v[3] == Approx(-44.65) );
  col.add_multiple(v, -5.5);

  col.weighted_add_multiple(v, w, -1.5);
  REQUIRE( v[0] == Approx(-2.65) );
  REQUIRE( v[2] == Approx(9.9) );
  REQUIRE( v[3] == Approx(-11.035) );

  REQUIRE( col.sum() == Approx(-4.8) );
  REQUIRE( col.mean() == Approx(-4.8/4.) );
  REQUIRE( col.l2_norm_sq() == Approx(74.28) );
}

TEST_CASE( "dense_column_types", "[column]" ) {
  double data_double[2] = {0., 1.};
  float data_float[2] = {0., 1.};
  int data_int[2] = {0., 1.};
  bool data_bool[2] = {false, true};

  DenseColumn<double> col_double(data_double, 2);
  DenseColumn<float> col_float(data_float, 2);
  DenseColumn<int> col_int(data_int, 2);
  DenseColumn<bool> col_bool(data_bool, 2);
  
  REQUIRE( col_double.mean() == 0.5 );
  REQUIRE( col_float.mean() == 0.5 );
  REQUIRE( col_int.mean() == 0.5 );
  REQUIRE( col_bool.mean() == 0.5 );

  vector<value_t> v(2, 1.0); v[1] = 2.5;
  REQUIRE( col_double.inner_product(v) == 2.5 );
  REQUIRE( col_float.inner_product(v) == 2.5 );
  REQUIRE( col_int.inner_product(v) == 2.5 );
  REQUIRE( col_bool.inner_product(v) == 2.5 );
}
