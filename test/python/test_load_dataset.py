import unittest
import blitzml
import ctypes
import numpy as np
from scipy import sparse as sp


class TestLoadDenseDataset(unittest.TestCase):

  def setUp(self):
    np.random.seed(0)
    self.n = 10
    self.d = 20
    self.b = np.random.randn(self.n)
    self.b[0] = 23.
    self.b[9] = 32.
    self.A = np.random.randn(self.n, self.d)
    self.A[:,0] = 0
    self.A[0,0] = 3.
    self.A[9,0] = 4.

  def test_load_dense_double(self):
    A = self.A.astype(ctypes.c_double)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_dense_dataset_double")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_dense_float(self):
    A = self.A.astype(ctypes.c_float)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_dense_dataset_float")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_dense_int(self):
    A = self.A.astype(ctypes.c_int)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_dense_dataset_int")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_dense_bool(self):
    A = self.A.astype(ctypes.c_bool)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_dense_dataset_bool")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 2 ** 0.5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_dense_default(self):
    A = np.abs(self.A).astype(ctypes.c_uint)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_dense_dataset_float")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)


class TestLoadSparseDataset(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    self.n = 11
    self.d = 21
    self.b = np.random.randn(self.n)
    self.b[0] = 23.
    self.b[9] = 32.
    A = np.random.randn(self.n, self.d)
    A[A < 0.5] = 0.
    A[:,0] = 0
    A[0,0] = -3.
    A[9,0] = 4.
    self.A = sp.csc_matrix(A)

  def test_load_sparse_double(self):
    A = self.A  
    A.data = A.data.astype(ctypes.c_double)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_double")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_double_lil(self):
    A = self.A
    A.data = A.data.astype(ctypes.c_double)
    A = A.tolil()
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_double")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_float(self):
    A = self.A  
    A.data = A.data.astype(ctypes.c_float)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_float")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_float_lil(self):
    A = self.A
    A.data = A.data.astype(ctypes.c_float)
    A = A.tolil()
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_float")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_int(self):
    A = self.A  
    A.data = A.data.astype(ctypes.c_int)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_int")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_int_lil(self):
    A = self.A
    A.data = A.data.astype(ctypes.c_int)
    A = A.tolil()
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_int")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_bool(self):
    A = self.A  
    A.data = A.data.astype(ctypes.c_bool)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_bool")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 2 ** 0.5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_bool_lil(self):
    A = self.A
    A.data = A.data.astype(ctypes.c_bool)
    A = sp.lil_matrix(A, dtype=A.dtype)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_bool")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 2 ** 0.5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_default(self):
    A = self.A  
    A.data = np.abs(A.data).astype(ctypes.c_uint)
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_float")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)

  def test_load_sparse_default_lil(self):
    A = self.A
    A.data = np.abs(A.data).astype(ctypes.c_uint)
    A = A.tolil()
    dataset = blitzml._dataset.Dataset(A, self.b)
    create_call = dataset.dataset_c_wrapper._create_func.__name__
    self.assertEqual(create_call, "BlitzML_new_sparse_dataset_float")
    col0_norm = dataset.compute_column_norm_in_C(0)
    self.assertAlmostEqual(col0_norm, 5)
    self.assertEqual(dataset.get_b_value_from_C(0), 23.)
    self.assertEqual(dataset.get_b_value_from_C(9), 32.)



