# coding: utf-8

from ._core import *

from scipy import sparse as sp
import abc

lib.BlitzML_delete_dataset.restype = None
lib.BlitzML_delete_dataset.argtypes = [pointer_t]

new_dataset_calls = [lib.BlitzML_new_sparse_dataset_float,
                     lib.BlitzML_new_sparse_dataset_double,
                     lib.BlitzML_new_sparse_dataset_int,
                     lib.BlitzML_new_sparse_dataset_bool,
                     lib.BlitzML_new_dense_dataset_float,
                     lib.BlitzML_new_dense_dataset_double,
                     lib.BlitzML_new_dense_dataset_int,
                     lib.BlitzML_new_dense_dataset_bool]
for call in new_dataset_calls:
  call.restype = pointer_t
  wrapper_call_name = "{}_wrapper".format(call.__name__)
  setattr(lib, wrapper_call_name, wrap_new_call(call, lib.BlitzML_delete_dataset))

lib.BlitzML_new_sparse_dataset_double.restype = pointer_t
lib.BlitzML_new_sparse_dataset_double.argtypes = [
 index_t_p, size_t_p, double_t_p, index_t, index_t, size_t, value_t_p, index_t]

lib.BlitzML_new_sparse_dataset_float.restype = pointer_t
lib.BlitzML_new_sparse_dataset_float.argtypes = [
  index_t_p, size_t_p, float_t_p, index_t, index_t, size_t, value_t_p, index_t]

lib.BlitzML_new_sparse_dataset_int.restype = pointer_t
lib.BlitzML_new_sparse_dataset_int.argtypes = [
    index_t_p, size_t_p, int_t_p, index_t, index_t, size_t, value_t_p, index_t]

lib.BlitzML_new_sparse_dataset_bool.restype = pointer_t
lib.BlitzML_new_sparse_dataset_bool.argtypes = [
   index_t_p, size_t_p, bool_t_p, index_t, index_t, size_t, value_t_p, index_t]

lib.BlitzML_new_dense_dataset_double.restype = pointer_t
lib.BlitzML_new_dense_dataset_double.argtypes = [
                              double_t_p, index_t, index_t, value_t_p, index_t]

lib.BlitzML_new_dense_dataset_float.restype = pointer_t
lib.BlitzML_new_dense_dataset_float.argtypes = [
                               float_t_p, index_t, index_t, value_t_p, index_t]

lib.BlitzML_new_dense_dataset_int.restype = pointer_t
lib.BlitzML_new_dense_dataset_int.argtypes = [
                                 int_t_p, index_t, index_t, value_t_p, index_t]

lib.BlitzML_new_dense_dataset_bool.restype = pointer_t
lib.BlitzML_new_dense_dataset_bool.argtypes = [
                                bool_t_p, index_t, index_t, value_t_p, index_t]

lib.BlitzML_column_norm.restype = value_t
lib.BlitzML_column_norm.argtypes = [pointer_t, index_t]

lib.BlitzML_b_value_j.restype = value_t
lib.BlitzML_b_value_j.argtypes = [pointer_t, index_t]


class Dataset(object):

  def __init__(self, A, b):
    self.A = A
    self.b = b
    if sp.issparse(A):
      self.dataset_c_wrapper = self.load_sparse_dataset()
    else:
      self.dataset_c_wrapper = self.load_dense_dataset()


  @property
  def c_pointer(self):
    return self.dataset_c_wrapper.c_pointer


  def load_sparse_dataset(self):
    if not sp.isspmatrix_csc(self.A):
      if self.A.nnz > data_copy_warning_cutoff:
        msg = ("Copying sparse matrix to scipy.sparse.csc_matrix. "
               "To avoid copying, provide design "
               "matrix as a {} matrix.")
        try:
          msg = msg.format(self.A.best_format)
        except:
          msg = msg.format("CSC")
        warn(msg)
      self.A = sp.csc_matrix(self.A, dtype=self.A.dtype)

    if self.A.dtype == np.dtype(double_t):
      self.A_data, A_data_c = data_as(self.A.data, double_t_p)
      call_c = lib.BlitzML_new_sparse_dataset_double_wrapper
    elif self.A.dtype == np.dtype(int_t):
      self.A_data, A_data_c = data_as(self.A.data, int_t_p)
      call_c = lib.BlitzML_new_sparse_dataset_int_wrapper
    elif self.A.dtype == np.dtype(bool_t):
      self.A_data, A_data_c = data_as(self.A.data, bool_t_p)
      call_c = lib.BlitzML_new_sparse_dataset_bool_wrapper
    else:
      self.A_data, A_data_c = data_as(self.A.data, float_t_p)
      call_c = lib.BlitzML_new_sparse_dataset_float_wrapper

    self.A_indices, A_indices_c = data_as(self.A.indices, index_t_p)
    self.A_indptr, A_indptr_c = data_as(self.A.indptr, size_t_p)
    A_args_c = [A_indices_c, A_indptr_c, A_data_c]

    c_args = A_args_c + self.shape_args() + self.b_args()
    ret = call_c(*c_args)   
    return ret


  def load_dense_dataset(self):
    assert type(self.A) == np.ndarray, "For dense datasets, matrix must be of type numpy.ndarray"
    if not self.A.flags.f_contiguous:
      nnz = np.prod(self.A.shape)
      if nnz > data_copy_warning_cutoff:
        msg = ("Copying data to F-contiguous array. "
               "To avoid copying, pass design matrix in {} format.")
        try:
          msg = msg.format(self.A.best_format)
        except:
          msg = msg.format("F-contiguous")
        warn(msg)
      self.A = np.asfortranarray(self.A)

    if self.A.dtype == np.dtype(double_t):
      self.A_data, A_data_c = data_as(self.A, double_t_p)
      call_c = lib.BlitzML_new_dense_dataset_double_wrapper
    elif self.A.dtype == np.dtype(int_t):
      self.A_data, A_data_c = data_as(self.A, int_t_p)
      call_c = lib.BlitzML_new_dense_dataset_int_wrapper
    elif self.A.dtype == np.dtype(bool_t):
      self.A_data, A_data_c = data_as(self.A, bool_t_p)
      call_c = lib.BlitzML_new_dense_dataset_bool_wrapper
    else:
      self.A_data, A_data_c = data_as(self.A, float_t_p)
      call_c = lib.BlitzML_new_dense_dataset_float_wrapper

    c_args = [A_data_c] + self.shape_args() + self.b_args()
    return call_c(*c_args)


  def shape_args(self):
    height_c = index_t(self.A.shape[0])
    width_c = index_t(self.A.shape[1])
    if sp.isspmatrix(self.A):
      nnz_c = size_t(self.A.nnz)
      return [height_c, width_c, nnz_c]
    else:
      return [height_c, width_c]


  def b_args(self):
    _, b_c = data_as(self.b, value_t_p)
    length_b_c = index_t(len(self.b))
    return [b_c, length_b_c]


  def compute_column_norm_in_C(self, col):
    return lib.BlitzML_column_norm(self.c_pointer, index_t(col))

  def get_b_value_from_C(self, index):
    return lib.BlitzML_b_value_j(self.c_pointer, index_t(index))


