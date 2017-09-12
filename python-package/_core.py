# coding: utf-8

import os
import ctypes
import numpy as np 
from scipy import sparse as sp
import pickle
import glob

from ._lib_file_extension import get_lib_file_extension

def load_library():
  current_dir = os.path.abspath(os.path.dirname(__file__))
  glob_string = "**libblitzml.{}*".format(get_lib_file_extension())
  glob_pathname = os.path.join(current_dir, glob_string)
  for path in glob.glob(glob_pathname):
    try:
      return np.ctypeslib.load_library(path, "/")    
    except OSError:
      pass
  raise Exception("Could not load C library")
  
lib = load_library()

index_t = ctypes.c_int
value_t = ctypes.c_double
float_t = ctypes.c_float
double_t = ctypes.c_double
int_t = ctypes.c_int
unsigned_t = ctypes.c_uint
bool_t = ctypes.c_bool
long_t = ctypes.c_long
size_t = ctypes.c_size_t

pointer_t = ctypes.POINTER(ctypes.c_void_p)
value_t_p = ctypes.POINTER(value_t)
index_t_p = ctypes.POINTER(index_t)
float_t_p = ctypes.POINTER(float_t)
double_t_p = ctypes.POINTER(double_t)
int_t_p = ctypes.POINTER(int_t)
bool_t_p = ctypes.POINTER(bool_t)
size_t_p = ctypes.POINTER(size_t)
char_p = ctypes.c_char_p

lib.BlitzML_solve_problem.argtypes = [pointer_t, pointer_t, pointer_t, value_t_p, char_p, char_p]
lib.BlitzML_new_parameters.restype = pointer_t
lib.BlitzML_new_parameters.argtypes = [value_t_p, size_t]
lib.BlitzML_first_parameter.restype = value_t
lib.BlitzML_first_parameter.argtypes = [pointer_t]
lib.BlitzML_delete_parameters.restype = None
lib.BlitzML_delete_parameters.argtypes = [pointer_t]
lib.BlitzML_delete_solver.restype = None
lib.BlitzML_delete_solver.argtypes = [pointer_t]
lib.BlitzML_set_tolerance.restype = None
lib.BlitzML_set_tolerance.argtypes = [pointer_t, value_t]
lib.BlitzML_tolerance.restype = value_t
lib.BlitzML_tolerance.argtypes = []
lib.BlitzML_set_max_time.restype = None
lib.BlitzML_set_max_time.argtypes = [pointer_t, value_t]
lib.BlitzML_max_time.restype = value_t
lib.BlitzML_max_time.argtypes = []
lib.BlitzML_set_min_time.restype = None
lib.BlitzML_set_min_time.argtypes = [pointer_t, value_t]
lib.BlitzML_min_time.restype = value_t
lib.BlitzML_min_time.argtypes = []
lib.BlitzML_set_verbose.restype = None
lib.BlitzML_set_verbose.argtypes = [pointer_t, bool_t]
lib.BlitzML_verbose.restype = bool_t
lib.BlitzML_verbose.argtypes = []
lib.BlitzML_set_use_screening.restype = None
lib.BlitzML_set_use_screening.argtypes = [pointer_t, bool_t]
lib.BlitzML_use_screening.restype = bool_t
lib.BlitzML_use_screening.argtypes = []
lib.BlitzML_set_use_working_sets.restype = None
lib.BlitzML_set_use_working_sets.argtypes = [pointer_t, bool_t]
lib.BlitzML_use_working_sets.restype = bool_t
lib.BlitzML_use_working_sets.argtypes = []
lib.BlitzML_set_log_vectors.restype = None
lib.BlitzML_set_log_vectors.argtypes = [pointer_t, bool_t]
lib.BlitzML_log_vectors.restype = bool_t
lib.BlitzML_log_vectors.argtypes = []
lib.BlitzML_set_max_iterations.restype = None
lib.BlitzML_set_max_iterations.argtypes = [pointer_t, unsigned_t]
lib.BlitzML_max_iterations.restype = unsigned_t
lib.BlitzML_max_iterations.argtypes = []

class CPointerWrapper(object):
  def __init__(self, c_pointer, create_func, delete_func):
    self.c_pointer = c_pointer
    self._create_func = create_func
    self._delete_c_obj = delete_func

  def __del__(self):
    self._delete_c_obj(self.c_pointer)

def wrap_new_call(func, delete_func):
  def new_func(*args, **kwargs):
    obj = func(*args, **kwargs)
    obj_wrapper = CPointerWrapper(obj, func, delete_func)
    return obj_wrapper
  return new_func

lib.BlitzML_new_parameters_wrap = wrap_new_call(lib.BlitzML_new_parameters,
                                                lib.BlitzML_delete_parameters)

new_solver_calls = [lib.BlitzML_new_sparse_linear_solver,
                    lib.BlitzML_new_lasso_solver,
                    lib.BlitzML_new_sparse_logreg_solver]
for call in new_solver_calls:
  call.argtypes = []
  call.restype = pointer_t
  wrapper_call_name = "{}_wrapper".format(call.__name__)
  setattr(lib, wrapper_call_name, wrap_new_call(call, lib.BlitzML_delete_solver))
  

_module_vars = {"warnings_suppressed": False}

def add_line_breaks(message, length=0):
  max_chars = 79
  result = []
  for w in message.split():
    length += len(w) + 1
    if length > max_chars:
      result.append("\n")
      length = len(w) + 1
    result.append(w + " ")
  return "".join(result).strip()

def print_if_not_suppressed(message):
  if not _module_vars["warnings_suppressed"]:
    message = add_line_breaks(message)
    print(message)
    
def warn(message):
  print_if_not_suppressed("Warning: {}".format(message))

def value_error(message):
  raise ValueError(add_line_breaks(message, length=12))

def suppress_warnings():
  _module_vars["warnings_suppressed"] = True

def unsuppress_warnings():
  _module_vars["warnings_suppressed"] = False

data_copy_warning_cutoff = 1e7


def data_as(obj, ctypes_type):
  if obj.dtype != np.dtype(ctypes_type):
    nnz = np.prod(obj.shape)
    if nnz > data_copy_warning_cutoff:
      msg = ("Copying numpy.array of size {:d} from type {} to {}. "
             "Refer to documentation for tips on avoiding copying.")
      msg = msg.format(nnz, obj.dtype, ctypes_type.__name__)
      warn(message)
    obj = obj.astype(ctypes_type)
  return (obj, obj.ctypes.data_as(ctypes_type))


def decode_c_char_array(c_char_array):
  return c_char_array.value.decode()


class Problem(object):

  def _set_solver_options(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, "_{}".format(k), v)

  def _set_parameters(self, parameters):
    num_params_c = size_t(len(parameters))
    self._parameters, params_array_c = data_as(parameters, value_t_p)
    _parameters_c_wrap = lib.BlitzML_new_parameters_wrap(params_array_c, num_params_c)
    return _parameters_c_wrap

  def _set_result(self, result):
    self._result, result_c = data_as(result, value_t_p)
    return result_c

  def _setup_solution_status(self):
    self._solution_status = ctypes.create_string_buffer(b"", 64)
    return self._solution_status

  def _set_log_directory(self, log_directory):
    if log_directory is None:
      log_directory = ""
    check_log_directory(log_directory)
    self._log_directory = os.path.join(log_directory, "")
    return char_p(self._log_directory.encode("utf8"))

  def _format_solution_status(self):
    self._solution_status = decode_c_char_array(self._solution_status)

  @property
  def _stopping_tolerance(self):
    """Stopping tolerance for solve.  Optimization terminates if
        duality_gap / objective_value < stopping_tolerance .

    Default is 1e-3.
    """
    return lib.BlitzML_tolerance(self._solver_c_wrap.c_pointer)

  @_stopping_tolerance.setter
  def _stopping_tolerance(self, value):
    lib.BlitzML_set_tolerance(self._solver_c_wrap.c_pointer, value_t(value))

  @property
  def _max_time(self):
    """Time limit in seconds for solve call. If stopping tolerance is not 
    reached, optimization terminates after this number of seconds.  Default is 
    1 year.
    """
    return lib.BlitzML_max_time(self._solver_c_wrap.c_pointer)

  @_max_time.setter
  def _max_time(self, value):
    lib.BlitzML_set_max_time(self._solver_c_wrap.c_pointer, value_t(value))

  @property
  def _max_iterations(self):
    """Iterations limit for solve call. If stopping tolerance is not reached 
    after this number of iterations, optimization terminates.  Default is 
    100000.
    """
    return lib.BlitzML_max_iterations(self._solver_c_wrap.c_pointer)

  @_max_iterations.setter
  def _max_iterations(self, value):
    lib.BlitzML_set_max_iterations(self._solver_c_wrap.c_pointer, unsigned_t(value))

  @property
  def _min_time(self):
    """Minimum time in seconds for solve call.  Optimization continues until 
    this amount of time passes, even after reaching stopping tolerance.  
    Default is zero.
    """
    return lib.BlitzML_min_time(self._solver_c_wrap.c_pointer)

  @_min_time.setter
  def _min_time(self, value):
    lib.BlitzML_set_min_time(self._solver_c_wrap.c_pointer, value_t(value))

  @property
  def _verbose(self):
    """If True, algorithm prints convergence info (such as objective value) to
    stdout after each iteration during solve call.  Default is False.
    """
    return lib.BlitzML_verbose(self._solver_c_wrap.c_pointer)

  @_verbose.setter
  def _verbose(self, value):
    lib.BlitzML_set_verbose(self._solver_c_wrap.c_pointer, bool_t(value))

  @property
  def _use_working_sets(self):
    """If True, BlitzML solves the problem using working sets.  Default is 
    True.  Setting to False likely results in slower optimization.
    """
    return lib.BlitzML_use_working_sets(self._solver_c_wrap.c_pointer)

  @_use_working_sets.setter
  def _use_working_sets(self, value):
    lib.BlitzML_set_use_working_sets(self._solver_c_wrap.c_pointer, bool_t(value))

  @property
  def _use_screening(self):
    """If True, BlitzML solves the problem using safe screening.  Default is 
    True.  Setting to False may result in slower optimization.
    """
    return lib.BlitzML_use_screening(self._solver_c_wrap.c_pointer)

  @_use_screening.setter
  def _use_screening(self, value):
    lib.BlitzML_set_use_screening(self._solver_c_wrap.c_pointer, bool_t(value))

  @property
  def _log_vectors(self):
    """If True, BlitzML solves the problem using working sets.  Default is 
    True.  Setting to False likely results in slower optimization.
    """
    return lib.BlitzML_log_vectors(self._solver_c_wrap.c_pointer)

  @_log_vectors.setter
  def _log_vectors(self, value):
    lib.BlitzML_set_log_vectors(self._solver_c_wrap.c_pointer, bool_t(value))


class BlitzMLSolution(object):
  def __init__(self, weights, bias, dual_solution, 
               status, duality_gap, objective_value):
    self._weights = weights
    self._bias = bias
    self._dual_solution = dual_solution
    self._objective_value = objective_value
    self._duality_gap = duality_gap
    self._solution_status = status

  @property
  def weights(self):
    """Array of model's weight values."""
    return self._weights

  @weights.setter
  def weights(self, value):
    self._weights = weights

  @property
  def bias(self):
    """Value of model's bias term."""
    return self._bias

  @bias.setter
  def bias(self, value):
    self._bias = bias

  @property
  def dual_solution(self):
    """Dual solution to optimization problem."""
    return self._dual_solution

  @dual_solution.setter
  def dual_solution(self, value):
    self._dual_solution = dual_solution

  @property
  def objective_value(self):
    """Objective value of solution."""
    return self._objective_value

  @objective_value.setter
  def objective_value(self, value):
    self._objective_value = objective_value

  @property
  def duality_gap(self):
    """Duality gap between primal and dual solutions."""
    return self._duality_gap

  @duality_gap.setter
  def duality_gap(self, value):
    self._duality_gap = duality_gap

  @property
  def solution_status(self):
    """Status of Blitz algorithm upon returning model."""
    return self._solution_status

  @solution_status.setter
  def solution_status(self, value):
    self._solution_status = solution_status

  def save(self, filepath):
    """Save model to disk.

    Parameters
    ----------
    filepath : string
      Locaiton to save solution.
    """
    with open(filepath, "wb") as outfile:
      pickle.dump(self, outfile)

  def _compute_A_times_weights(self, A):
    if sp.issparse(A):
      result = A * np.mat(self.weights).T + self.bias
      return np.array(result).flatten()
    else:
      return np.dot(A, self.weights) + self.bias


class RegressionSolution(BlitzMLSolution):
  def predict(self, A):
    """Predict label values from feature vectors.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix to evaluate predictions for.

    Returns
    -------
    predictions : numpy.ndarray of floats
    """
    return self._compute_A_times_weights(A)


class ClassificationSolution(BlitzMLSolution):
  def predict(self, A):
    """Predict label values from feature vectors.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix to evaluate predictions for.

    Returns
    -------
    predictions : numpy.ndarray with values +/-1.
    """
    preds = self._compute_A_times_weights(A)
    preds[preds < 0] = -1.
    preds[preds >= 0] = 1.
    return preds
 

def load_solution(filepath):
  """Load BlitzMLSolution from disk.

  Parameters
  ----------
  filepath : string
    Path to saved BlitzMLSolution.

  Returns
  -------
  solution : BlitzMLSolution
  """
  with open(filepath) as f:
    solution = pickle.load(f)
  return solution


def check_log_directory(dir_path):
  if dir_path == "":
    return
  if not os.path.exists(dir_path):
    warn("Provided log directory does not exist")
  elif os.listdir(dir_path):
    warn("Starting optimization with non-empty log directory")


def check_classification_labels(b, min_b=None, max_b=None):
  if min_b is None:
    min_b = min(b)
  if max_b is None:
    max_b = max(b)
  if min_b < -1.0:
    msg = ("Labels vector conatins values less than -1.0, which is "
           "unusual. Typically labels are +/-1 for classification.")
    warn(msg)
  max_b = max(b)
  if max_b > 1.0:
    msg = ("Labels vector conatins values greater than 1.0, which is "
           "unusual. Typically labels are +/-1 for classification.")
    warn(msg)
  if min_b >= 0.:
    msg = ("Labels vector contains no values less than zero, which is "
           "unusual. Use -1.0 to label negative training instances.")
    warn(msg)
  if max_b <= 0.:
    msg = ("Labels vector contains no values greater than zero, which is "
           "unusual. Use 1.0 to label positive training instances.")
    warn(msg)


