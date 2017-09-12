#include "catch.hpp"

#include <blitzml/base/common.h>
#include <blitzml/dataset/sparse_dataset.h>
#include <blitzml/dataset/dense_dataset.h>
#include <blitzml/dataset/sparse_binary_dataset.h>

using std::vector;

using namespace BlitzML;

TEST_CASE( "test_sparse_dataset", "[dataset]" ) {
  vector<index_t> indices;
  vector<float> data;
  vector<size_t> indptr;
  indptr.push_back(0);

  // Column 1
  indptr.push_back(0); 

  // Column 2
  indices.push_back(2); data.push_back(2.5);
  indices.push_back(4); data.push_back(-0.5);
  indptr.push_back(2);

  // Column 3
  indptr.push_back(2);

  // Column 4
  indices.push_back(0); data.push_back(-0.5);
  indices.push_back(4); data.push_back(0.3);
  indptr.push_back(4);

  // Column 5
  indptr.push_back(4);

  // Column 6
  indices.push_back(1); data.push_back(2.0);
  indices.push_back(2); data.push_back(2.0);
  indices.push_back(3); data.push_back(2.0);
  indptr.push_back(7);

  vector<value_t> b(5, 0.);
  b[0] = 1.0;
  b[2] = 2.0;
  b[4] = -0.5;

  SparseDataset<float> ds(&indices[0], &indptr[0], &data[0], 
                          5, 6, data.size(), &b[0], 5);

  vector<value_t> v(5, 0.);
  v[0] = 2.0;
  v[3] = 1.0;
  v[4] = 0.2;
  value_t val = ds.column(3)->inner_product(v);
  REQUIRE( val == Approx( -0.94 ) );

  val = ds.column(0)->inner_product(v);
  REQUIRE( val == 0. );

  val = ds.column(1)->inner_product(v);
  REQUIRE( val == Approx( -0.1 ) );

  REQUIRE( ds.any_columns_overlapping_in_submatrix(0, 4) == true );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(2, 4) == false );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(0, 3) == false );

  REQUIRE( ds.b_value(0) == 1.0 );
  REQUIRE( ds.b_value(1) == 0.0 );
  REQUIRE( ds.b_value(4) == -0.5 );

  REQUIRE( ds.num_rows() == 5 );
  REQUIRE( ds.num_cols() == 6 );
  REQUIRE( ds.nnz() == 7 );
}

TEST_CASE( "test_sparse_dataset_types", "[dataset]" ) {
  vector<double> data_double(2, 1.0);
  vector<float> data_float(2, 1.0);
  vector<int> data_int(2, 1);
  bool data_bool[2] = {true, true};

  vector<index_t> indices;
  indices.push_back(0); indices.push_back(1);

  vector<size_t> indptr;
  indptr.push_back(0); indptr.push_back(2);

  vector<value_t> b(2, 0.);

  SparseDataset<double> ds_double(&indices[0], &indptr[0], &data_double[0],
                                  2, 1, 2, &b[0], 2);
  SparseDataset<float> ds_float(&indices[0], &indptr[0], &data_float[0],
                                  2, 1, 2, &b[0], 2);
  SparseDataset<int> ds_int(&indices[0], &indptr[0], &data_int[0],
                                  2, 1, 2, &b[0], 2);
  SparseDataset<bool> ds_bool(&indices[0], &indptr[0], &data_bool[0],
                                  2, 1, 2, &b[0], 2);

  vector<value_t> v;
  v.push_back(-2.0);
  v.push_back(2.6);

  REQUIRE( ds_double.column(0)->inner_product(v) == Approx( 0.6 ) );
  REQUIRE( ds_float.column(0)->inner_product(v) == Approx( 0.6 ) );
  REQUIRE( ds_int.column(0)->inner_product(v) == Approx( 0.6 ) );
  REQUIRE( ds_bool.column(0)->inner_product(v) == Approx( 0.6 ) );
}

TEST_CASE( "test_dense_dataset", "[datset]" ) {
  double data[12] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
  value_t b[4] = {1., 1., -3., 1.}; 
  DenseDataset<double> ds(data, 4, 3, b, 4);

  value_t val = ds.column(0)->l2_norm_sq();
  REQUIRE( ds.column(0)->l2_norm_sq() == Approx( 30. ) );

  vector<value_t> vec(4, 0.0);
  vec[3] = -1.0;
  vec[1] = -0.5;
  REQUIRE( ds.column(2)->inner_product(vec) == Approx( -17. ) );

  REQUIRE( ds.any_columns_overlapping_in_submatrix(0, 2) == true );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(0, 3) == true );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(0, 1) == false );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(1, 2) == false );

  REQUIRE( ds.b_value(2) == -3. );
  REQUIRE( ds.b_value(0) == 1. );
  REQUIRE( ds.b_value(3) == 1. );

  REQUIRE( ds.num_rows() == 4 );
  REQUIRE( ds.num_cols() == 3 );
  REQUIRE( ds.nnz() == 12 );
}

TEST_CASE( "test_dense_dataset_types", "[dataset]" ) {
  double data_double[2] = {1., 1.};
  float data_float[2] = {1., 1.};
  int data_int[2] = {1., 1.};
  bool data_bool[2] = {true, true};

  DenseDataset<double> ds_double(data_double, 2, 1, NULL, 0);
  DenseDataset<float> ds_float(data_float, 2, 1, NULL, 0);
  DenseDataset<int> ds_int(data_int, 2, 1, NULL, 0);
  DenseDataset<bool> ds_bool(data_bool, 2, 1, NULL, 0);

  vector<value_t> vec;
  vec.push_back(-2.6); vec.push_back(0.3);
  REQUIRE( ds_double.column(0)->inner_product(vec) == Approx(-2.3) );
  REQUIRE( ds_float.column(0)->inner_product(vec) == Approx(-2.3) );
  REQUIRE( ds_int.column(0)->inner_product(vec) == Approx(-2.3) );
  REQUIRE( ds_bool.column(0)->inner_product(vec) == Approx(-2.3) );
}

TEST_CASE( "test_binary_dataset", "[datset]" ) {
  index_t indices[8] = {0, 2, 3, 0, 3, 1, 2, 4};
  size_t indptr[5] = {0, 0, 3, 5, 8};
  value_t b[5] = {0., 1.0, 0., -2.5, 3.0};
  SparseBinaryDataset ds(indices, indptr, 5, 4, 8, b, 5);

  vector<value_t> vec;
  vec.push_back(0.1); vec.push_back(0.2); vec.push_back(-0.3); 
  vec.push_back(-0.4); vec.push_back(0.5);

  REQUIRE( ds.column(0)->inner_product(vec) == 0. );
  REQUIRE( ds.column(3)->inner_product(vec) == Approx(0.4) );

  REQUIRE( ds.any_columns_overlapping_in_submatrix(2, 4) == false );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(3, 4) == false );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(1, 4) == true );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(0, 2) == false );
  REQUIRE( ds.any_columns_overlapping_in_submatrix(0, 3) == true );

  REQUIRE( ds.b_value(0) == 0. );
  REQUIRE( ds.b_value(4) == 3. );
  REQUIRE( ds.b_value(3) == -2.5 );

  REQUIRE( ds.num_rows() == 5 );
  REQUIRE( ds.num_cols() == 4 );
  REQUIRE( ds.nnz() == 8 );
}

TEST_CASE( "test_dataset_general", "[dataset]" ) {
  double data[12] = {1., -0.3, 0., 2.1, 5., 1.0, 7., 8., 9., 0., -1., 12.};
  value_t b[4] = {1., 1., -3., 1.}; 
  DenseDataset<double> ds(data, 4, 3, b, 4);

  vector<value_t> vec(4, 0.);
  vec[0] = -1.1;
  vec[1] = -2.1;
  vec[2] = 1.8;
  vec[3] = 10.1;

  vector<value_t> result(2, 0.);

  ds.contiguous_submatrix_multiply(vec, &result[0], 1, 3);
  REQUIRE( result[0] == Approx(85.8) );
  REQUIRE( result[1] == Approx(109.5) );
}


