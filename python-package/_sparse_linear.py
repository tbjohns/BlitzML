# coding: utf-8

from ._core import *
from ._dataset import Dataset


lib.BlitzML_sparse_linear_solver_compute_max_l1_penalty.argtypes = [pointer_t, pointer_t, pointer_t]
lib.BlitzML_sparse_linear_solver_compute_max_l1_penalty.restype = value_t

class SparseLinearProblem(Problem):
  def __init__(self, A, b):
    self._check_data_inputs(A, b)
    self._dataset = Dataset(A, b) 
    self._set_solver_c()

  def _set_solver_c(self):
    self._solver_c_wrap = lib.BlitzML_new_sparse_linear_solver_wrapper()

  def _check_data_inputs(self, A, b):
    if not sp.isspmatrix(A) and type(A) != np.ndarray:
      msg = ("Design matrix of type {} not allowed. Type must be "
             "numpy.ndarray or scipy sparse matrix.").format(type(A).__name__)
      value_error(msg)
    if type(b) != np.ndarray:
      msg = ("Labels vector of type {} not allowed. Type must be "
             "numpy.ndarray.").format(type(b).__name__)
      value_error(msg)
    if A.ndim != 2:
      msg = ("Design matrix with shape {} not allowed. "
             "Matrix must be 2d.").format(A.shape)
      value_error(msg)
    if b.ndim != 1:
      msg = ("Labels vector with shape {} not allowed. " 
             "Labels must be 1d.").format(b.shape)
      value_error(msg)
    if A.shape[0] != len(b):
      msg = ("Dimension mismatch between matrix shape {} and labels shape {}. "
             "Length of labels vector must equal number of rows in "
             "design matrix.").format(A.shape, b.shape)
      value_error(msg)

  def compute_max_l1_penalty(self, include_bias_term=True):
    """Compute the smallest L1 penalty parameter for which all weights
    in the solution equal zero.

    Parameters
    ----------
    include_bias_term : bool, optional
      Whether to include an unregularized bias term in the model.  Default
      is True.

    Returns
    -------
    max_l1_penalty : float
    """
    parameters = np.zeros(4)
    parameters[2] = float(include_bias_term)
    parameters[3] = self._loss_index
    parameters_c_wrap = self._set_parameters(parameters)

    ret = lib.BlitzML_sparse_linear_solver_compute_max_l1_penalty(
                                              self._solver_c_wrap.c_pointer, 
                                              self._dataset.c_pointer, 
                                              parameters_c_wrap.c_pointer)
    return ret
    

  def solve(self, l1_penalty, include_bias_term=True, initial_weights=None, 
            stopping_tolerance=1e-3, max_time=3.154e+7, min_time=0., 
            max_iterations=100000, verbose=False, log_directory=None):
    r"""Minimizes the objective

    .. math::

       \sum_i L(a_i^T w, b_i) + \texttt{l1\_penalty} ||w||_1 ,

    where L is the problem's loss function.

    Parameters
    ----------
    l1_penalty : float > 0
      Regularization parameter for L1 norm penalty on weights. When this value
      is larger, the solution generally contains more zero entries and Blitz
      generally completes optimization faster.

    include_bias_term : bool, optional
      Whether to include an unregularized bias parameter in the model.  
      Default is True.

    initial_weights : numpy.ndarray of length d, optional
      Initial weights to warm-start optimization.  The solver requires
      less time if initialized to a good approximate solution. Default is None.

    stopping_tolerance : float, optional
      Stopping tolerance for solver.  Optimization terminates if
      (duality_gap / objective_value) is less than stopping_tolerance.
      Default is 1e-3.

    max_time : float, optional
      Time limit in seconds for solving. If stopping tolerance is not reached, 
      optimization terminates after this number of seconds.  Default is 1 year.

    min_time : float, optional
      Minimum time in seconds for solving.  Optimization continues until 
      this amount of time passes, even after reaching stopping tolerance.  
      Default is zero.
    
    max_iterations : int, optional
      Iterations limit for algorithm. If stopping tolerance is not reached 
      after this number of iterations, optimization terminates.  Default is 
      100000.

    verbose : bool, optional
      Whether to print information, such as objective value, to sys.stdout 
      during optimization. Default is False.

    log_directory : string, optional
      Path to existing directory for Blitz to log time and objective value 
      information.  This directory should be empty prior to solving. Default
      is None.

      
    Returns
    -------
    solution : BlitzMLSolution 
    """
    option_names = ["stopping_tolerance", "max_time", "min_time",
                    "max_iterations", "verbose"] 
    options = {}
    for name in option_names:
      options[name] = locals()[name]
    self._set_solver_options(**options)

    parameters = np.zeros(4)
    parameters[0] = l1_penalty
    parameters[2] = float(include_bias_term)
    parameters[3] = self._loss_index
    parameters_c_wrap = self._set_parameters(parameters)

    num_examples, num_features = self._dataset.A.shape
    result = np.zeros(num_features + 1 + num_examples + 2)
    if initial_weights is not None:
      result[0:num_features] = initial_weights
    result_c = self._set_result(result)

    solution_status_c = self._setup_solution_status()

    log_dir_c = self._set_log_directory(log_directory)

    lib.BlitzML_solve_problem(self._solver_c_wrap.c_pointer, 
                              self._dataset.c_pointer, 
                              parameters_c_wrap.c_pointer, 
                              result_c, 
                              solution_status_c, 
                              log_dir_c)

    self._format_solution_status()

    weights = result[0:num_features]
    bias = result[num_features]
    dual_solution = result[1+num_features:1+num_features+num_examples]
    duality_gap = result[-2]
    objective_value = -result[-1]

    return self._solution_class(weights, 
                                bias, 
                                dual_solution, 
                                self._solution_status,
                                duality_gap, 
                                objective_value)


class LassoProblem(SparseLinearProblem):
  r"""Class for training sparse linear models with squared loss.
  The optimization objective is

  .. math::

    \tfrac{1}2 ||A w - b||^2 + \lambda ||w||_1 .

  Parameters
  ----------
  A : numpy.ndarray or scipy.sparse matrix
    n x d design matrix.

  b : numpy.ndarray
    Labels array of length n.


  Blitz tries its best to avoid copying data, but depending on the formats of A
  and b, Blitz may make a copy of these arrays. To avoid copying, follow these
  guidelines:

  * If A is dense (numpy.ndarray), define A as an `F-contiguous array
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.asfortranarray.html>`_.
    The dtype for A should match ctypes.c_double, ctypes.c_float, ctypes.c_int, 
    or ctypes.c_bool---that is, ``A.dtype == ctypes.c_double`` evaluates to 
    True, for example.

  * If A is sparse, define A as a scipy.sparse.csc_matrix. The dtype for
    A.indices should match type ctypes.c_int. The dtype for A.indptr should
    match type ctypes.c_size_t. BlitzML can work with double, float, int, and
    bool dtypes for A.data without copying.

  * The dtype for b should match type ctypes.c_double.

  BlitzML will print a warning when objects with more than 1e7 elements 
  are being copied.  To suppress warnings, use ``blitzml.suppress_warnings()``. 
  """

  @property
  def _solution_class(self):
    return LassoSolution

  @property
  def _loss_index(self):
    return 0.

  def _set_solver_c(self):
    self._solver_c_wrap = lib.BlitzML_new_lasso_solver_wrapper()


class SparseHuberProblem(SparseLinearProblem):
  r"""Class for training sparse linear models with huber loss.
  The optimization objective is

  .. math::

    \sum_i L(a_i^T w, b_i) + \lambda ||w||_1 ,  
 
  where
  
  .. math::

    L(a_i^T w, b_i) = \left\{
    \begin{array}{lll}
      \tfrac{1}2 (a_i^T w - b_i)^2 & & \text{if}\ |a_i^T w - b_i| < 1  \\[0.4em]
      a_i^T w - b_i - \onehalf     & & \text{if}\ a_i^T w - b_i \geq 1 \\[0.4em]
      b_i - a_i^T w - \onehalf     & & \text{otherwise.} 
    \end{array} \right.

  Here i indexes the ith row in A and ith entry in b.
  """
  @property
  def _solution_class(self):
    return SparseHuberSolution

  @property
  def _loss_index(self):
    return 1.


class SparseLogisticRegressionProblem(SparseLinearProblem):
  r"""Class for training sparse linear models with logistic loss.
  The optimization objective is

  .. math::

    \sum_i \log(1 + \exp(-b_i a_i^T w)) + \lambda ||w||_1 ,

  where i indexes the ith row in A and ith entry in b.  Each label should have
  value 1, -1, or somewhere in between.
  """
  @property
  def _solution_class(self):
    return SparseLogisticRegressionSolution

  @property
  def _loss_index(self):
    return 2.

  def _check_data_inputs(self, A, b):
    SparseLinearProblem._check_data_inputs(self, A, b)
    min_b = min(b)
    if min_b < -1.0:
      msg = ("Labels vector conatins values less than -1.0, which is "
             "not allowed for sparse logistic regression.")
      value_error(msg)
    max_b = max(b)
    if max_b > 1.0:
      msg = ("Labels vector contains values greater than 1.0, which is "
             "not allowed for sparse logistic regression.")
      value_error(msg)
    check_classification_labels(b, min_b, max_b)

  def _set_solver_c(self):
    self._solver_c_wrap = lib.BlitzML_new_sparse_logreg_solver_wrapper()


class SparseSquaredHingeProblem(SparseLinearProblem):
  r"""Class for training sparse linear models with squared hinge loss.
  The optimization objective is

  .. math::

    \sum_i \onehalf (1 - b_i a_i^T w)_+^2 + \lambda ||w||_1 ,

  where the "+" subscript denotes the rectifier function.  Each label should 
  have value 1 or -1.
  """
  @property
  def _solution_class(self):
    return SparseSquaredHingeSolution

  @property
  def _loss_index(self):
    return 3.

  def _check_data_inputs(self, A, b):
    SparseLinearProblem._check_data_inputs(self, A, b)
    check_classification_labels(b)


class SparseSmoothedHingeProblem(SparseLinearProblem):
  r"""Class for training sparse linear models with smoothed hinge loss.
  The optimization objective is

  .. math::

    \sum_i L(a_i^T w, b_i) + \lambda ||w||_1 ,  
 
  where
  
  .. math::

    L(a_i^T w, b_i) = \left\{
    \begin{array}{lll}
      \onehalf - b_i a_i^T w       && \text{if}\ b_i a_i^T w < 0       \\[0.4em]
      \onehalf (1 - b_i a_i^T w)^2 && \text{if}\ b_i a_i^T w \in [0,1) \\[0.4em]
      0                            && \text{otherwise.} 
    \end{array} \right.

  Each label should have value 1 or -1.
  """
  @property
  def _solution_class(self):
    return SparseSmoothedHingeSolution

  @property
  def _loss_index(self):
    return 4.

  def _check_data_inputs(self, A, b):
    SparseLinearProblem._check_data_inputs(self, A, b)
    check_classification_labels(b)


class LassoSolution(RegressionSolution):
  """Solution class for ``LassoProblem``."""

  def _compute_loss(self, Aw, b):
    loss = 0.5 * np.linalg.norm(Aw - b) ** 2
    return loss


class SparseHuberSolution(RegressionSolution):
  """Solution class for ``SparseHuberProblem``."""
  def _compute_loss(self, Aw, b):
    width = 1.
    residuals = Aw - b
    one_half_width_sq = 0.5 * width ** 2
    
    left = np.sum(-width * residuals[residuals < -width] - one_half_width_sq)
    residuals[residuals < -width] = 0.

    right = np.sum(width * residuals[residuals > width] - one_half_width_sq)
    residuals[residuals > width] = 0.

    middle = np.sum(residuals ** 2) * 0.5
    
    return left + middle + right


class SparseLogisticRegressionSolution(ClassificationSolution):
  """Solution class for ``SparseLogisticRegressionProblem``."""
  def predict_probabilities(self, A):
    """Predict probability values from feature vectors.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix to predict probabilities for.

    Returns
    -------
    predicted_probabilities : numpy.ndarray (length n with values in [0, 1]).
    """
    Aw = self._compute_A_times_weights(A)
    probs = 1 / (1 + np.exp(-Aw))
    return probs

  def _compute_loss(self, Aw, b):
    return np.sum(np.log1p(np.exp(-b * Aw)))


class SparseSquaredHingeSolution(ClassificationSolution):
  """Solution class for ``SparseSquaredHingeProblem``.  """

  def _compute_loss(self, Aw, b):
    loss_terms = 1 - b * Aw
    loss_terms[loss_terms < 0] = 0.
    return 0.5 * np.sum(loss_terms ** 2)


class SparseSmoothedHingeSolution(ClassificationSolution):
  """Solution class for ``SparseSquaredHingeProblem``.  """

  def _compute_loss(self, Aw, b):
    loss_terms = 1 - b * Aw
    loss_terms[loss_terms < 0] = 0.
    squared_examples = (loss_terms < 1)
    squared_loss = 0.5 * np.sum(loss_terms[squared_examples] ** 2)
    linear_examples = (loss_terms > 1)
    linear_loss = np.sum(loss_terms[linear_examples] - 0.5)
    return squared_loss + linear_loss


