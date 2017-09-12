# coding: utf-8

from ._core import *
from ._dataset import Dataset


lib.BlitzML_sparse_linear_solver_compute_max_l1_penalty.argtypes = [pointer_t, pointer_t, pointer_t]
lib.BlitzML_sparse_linear_solver_compute_max_l1_penalty.restype = value_t


class _SparseLinearProblem(Problem):
  def __init__(self, A, b):
    """
    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix.

    b : numpy.ndarray
      Labels array of length n.

    Notes
    -----
    Depending on the formats of A and b, BlitzML may make a copy of the data. To
    avoid data copying, follow these guidelines:
    * If A is dense (numpy.ndarray), define A as an F-contiguous array.
      The dtype for A should match ctypes.c_float, ctypes.c_double, or 
      ctypes.c_int, or ctypes.c_bool--for example, 
      A.dtype == numpy.dtypes(ctypes.c_double) evaluates to True.
    * If A is sparse, define A as a scipy.sparse.csc_matrix. The dtype for
      A.indices should match type ctypes.c_int. The dtype for A.indptr should
      match type ctypes.c_size_t. BlitzML can work with float, double, int, and
      bool dtypes for A.data without copying.
    * If the data dtype is some other type, BlitzML copies the data to type 
      float by default.
    * The dtype for b should match type ctypes.c_double.

    BlitzML will print a warning when objects with more than 1e7 elements 
    are begin copied.  To suppress warnings, use blitzml.suppress_warnings(). 
    """
    self._dataset = Dataset(A, b) 
    self._set_solver_c()

  def _set_solver_c(self):
    self._solver_c_wrap = lib.BlitzML_new_sparse_linear_solver_wrapper()

  def compute_max_l1_penalty(self, include_bias_term=True):
    """Compute the smallest l1 regularization penalty for which all weights
    in the solution equal zero.

    Parameters
    ----------
    include_bias_term : bool
      Whether to include an unregularized bias term in the model.  Default
      is True.

    Returns
    -------
    l1_penalty_max : float
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
            max_iterations=100000, verbose=False, _log_directory=None):
    """
    Minimizes the objective
       sum_i L(a_i^T w, b_i) + l1_penalty ||w||_1 ,

    where L is the problem's loss function.

    Parameters
    ----------
    l1_penalty : float > 0
      Regularization parameter for L1 norm penalty on weights.  Note: in
      general, Blitz completes optimization faster when this value is larger.

    include_bias_term (optional) : bool
      Whether to include an unregularized bias parameter in the model.  
      Default is true.

    initial_weights (optional) : iterable(float) of length A.shape[1]
      Initial weights to warm-start optimization.  The algorithm will 
      terminate after less time if initialized to a good approximate solution.

    stopping_tolerance (optional) : float
      Stopping tolerance for solve.  Optimization terminates if
          duality_gap / objective_value < stopping_tolerance .
      Default is 1e-3.

    max_time (optional) : float
      Time limit in seconds for solving. If stopping tolerance is not reached, 
      optimization terminates after this number of seconds.  Default is 1 year.

    min_time (optional) : float
      Minimum time in seconds for solving.  Optimization continues until 
      this amount of time passes, even after reaching stopping tolerance.  
      Default is zero.
    
    max_iterations (optional) : int
      Iterations limit for algorithm. If stopping tolerance is not reached 
      after this number of iterations, optimization terminates.  Default is 
      100000.

    verbose (optional) : bool
      Whether to print information, such as objective value, to stdout during
      optimization.

    _log_directory (optional) : string
      Path to existing directory for Blitz to log time and objective value 
      information.

      
    Returns
    -------
    problem solution : BlitzMLSolution 

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

    log_dir_c = self._set_log_directory(_log_directory)

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


class LassoProblem(_SparseLinearProblem):
  """Class for training sparse linear models with squared loss.

  The optimization objective is
    sum_i L(a_i^T w, b_i) + l1_penalty ||w||_1 ,

  where L(a_i^T w, b_i) = 0.5 (a_i^T w - b_i)^2 .  
  
  Here i is the ith row of design matrix A, while b_i is the ith element of b.
  """
  @property
  def _solution_class(self):
    return LassoSolution

  @property
  def _loss_index(self):
    return 0.

  def _set_solver_c(self):
    self._solver_c_wrap = lib.BlitzML_new_lasso_solver_wrapper()


class SparseHuberProblem(_SparseLinearProblem):
  """Class for training sparse linear models with huber loss.

  The optimization objective is
    sum_i L(a_i^T w, b_i) + l1_penalty ||w||_1 .

  To define L in this case, define r_i = b_i - a_i^T w.  Then

    L(a_i^T w, b_i) = 
        0.5 (r_i)^2   if |r_i| < 1,  
        r_i - 0.5     if r_i >= 1,
        -r_i - 0.5    otherwise.

  Here i is the ith row of design matrix A, and b_i is the ith element of b.
  """
  @property
  def _solution_class(self):
    return SparseHuberSolution

  @property
  def _loss_index(self):
    return 1.


class SparseLogisticRegressionProblem(_SparseLinearProblem):
  """Class for training sparse linear models with logistic loss.

  The optimization objective is
    sum_i L(a_i^T w, b_i) + l1_penalty ||w||_1 ,

  where L(a_i^T w, b_i) = log(1 + exp(-b_i a_i^T w)).

  Here i is the ith row of design matrix A, and b_i is the ith element of b.
  """
  @property
  def _solution_class(self):
    return SparseLogisticRegressionSolution

  @property
  def _loss_index(self):
    return 2.

  def _set_solver_c(self):
    self._solver_c_wrap = lib.BlitzML_new_sparse_logreg_solver_wrapper()


class SparseSquaredHingeProblem(_SparseLinearProblem):
  """Class for training sparse linear models with squared hinge loss.

  The optimization objective is
    sum_i L(a_i^T w, b_i) + l1_penalty ||w||_1 ,

  where L(a_i^T w, b_i) = 
          0.5 (1 - b_i a_i^T w)^2   if  b_i a_i^T w < 1,
          0                         otherwise.

  Here i is the ith row of design matrix A, and b_i is the ith element of b.
  """
  @property
  def _solution_class(self):
    return SparseSquaredHingeSolution

  @property
  def _loss_index(self):
    return 3.


class SparseSmoothedHingeProblem(_SparseLinearProblem):
  """Class for training sparse linear models with smoothed hinge loss.

  The optimization objective is
    sum_i L(a_i^T w, b_i) + l1_penalty ||w||_1 ,

  where L(a_i^T w, b_i) = 
          0.5 - b_i a_i^T w         if b_i a_i^T w < 0,
          0.5 (1 - b_i a_i^T w)^2   if  0 <= b_i a_i^T w < 1,
          0                         otherwise.

  Here i is the ith row of design matrix A, and b_i is the ith element of b.
  """
  @property
  def _solution_class(self):
    return SparseSmoothedHingeSolution

  @property
  def _loss_index(self):
    return 4.


class LassoSolution(RegressionSolution):
  """Solution object for LassoProblem."""

  def evaluate_loss(self, A, b):
    """Computes the value
         sum_i L(a_i^T w, b_i) ,

    where L(a_i^T w, b_i) = 0.5 * (a_i^T w - b_i)^2 .

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix for evaluating loss.

    b : numpy.ndarray
      Labels array for evaluating loss.

    Returns
    -------
    loss : float
    """
    Aw = self._compute_A_times_weights(A)
    loss = 0.5 * np.linalg.norm(Aw - b) ** 2
    return loss


class SparseHuberSolution(RegressionSolution):
  """Solution object for SparseHuberProblem.
  """
  def evaluate_loss(self, A, b):
    """Computes the value
         sum_i L(a_i^T w, b_i) ,

    To define L in this case, define r_i = b_i - a_i^T w.  Then

      L(a_i^T w, b_i) = 
          0.5 (r_i)^2   if |r_i| < 1,  
          r_i - 0.5     if r_i >= 1,
          -r_i - 0.5    otherwise.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix for evaluating loss.

    b : numpy.ndarray
      Labels array for evaluating loss.

    Returns
    -------
    loss : float
    """
    width = 1.
    residuals = self._compute_A_times_weights(A) - b
    one_half_width_sq = 0.5 * width ** 2
    
    left = np.sum(-width * residuals[residuals < -width] - one_half_width_sq)
    residuals[residuals < -width] = 0.

    right = np.sum(width * residuals[residuals > width] - one_half_width_sq)
    residuals[residuals > width] = 0.

    middle = np.sum(residuals ** 2) * 0.5
    
    return left + middle + right


class SparseLogisticRegressionSolution(ClassificationSolution):
  """Solution object for SparseLogisticRegressionProblem.
  """
  def predict_probabilities(self, A):
    Aw = self._compute_A_times_weights(A)
    probs = 1 / (1 + np.exp(-Aw))
    return probs

  def evaluate_loss(self, A, b):
    """Computes the value
         sum_i L(a_i^T w, b_i) ,

    where L(a_i^T w, b_i) = log(1 + exp(-b_i a_i^T w)).

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix for evaluating loss.

    b : numpy.ndarray
      Labels array for evaluating loss.

    Returns
    -------
    loss : float
    """
    bAw = b * self._compute_A_times_weights(A)
    return np.sum(np.log1p(np.exp(-bAw)))


class SparseSquaredHingeSolution(ClassificationSolution):
  """Solution object for SparseSquaredHingeProblem.
  """
  def evaluate_loss(self, A, b):
    """Computes the value
         sum_i L(a_i^T w, b_i) ,

    where L(a_i^T w, b_i) = 
            0.5 (1 - b_i a_i^T w)^2   if  b_i a_i^T w < 1,
            0                         otherwise.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix for evaluating loss.

    b : numpy.ndarray
      Labels array for evaluating loss.

    Returns
    -------
    loss : float
    """
    loss_terms = 1 - b * self._compute_A_times_weights(A)
    loss_terms[loss_terms < 0] = 0.
    return 0.5 * np.sum(loss_terms ** 2)


class SparseSmoothedHingeSolution(ClassificationSolution):
  """Solution object for SparseSmoothedHingeProblem.
  """
  def evaluate_loss(self, A, b):
    """Computes the value
         sum_i L(a_i^T w, b_i) ,

    where L(a_i^T w, b_i) = 
            0.5 - b_i a_i^T w         if b_i a_i^T w < 0,
            0.5 (1 - b_i a_i^T w)^2   if  0 <= b_i a_i^T w < 1,
            0                         otherwise.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
      n x d design matrix for evaluating loss.

    b : numpy.ndarray
      Labels array for evaluating loss.

    Returns
    -------
    loss : float
    """
    loss_terms = 1 - b * self._compute_A_times_weights(A)
    loss_terms[loss_terms < 0] = 0.
    squared_examples = (loss_terms < 1)
    squared_loss = 0.5 * np.sum(loss_terms[squared_examples] ** 2)
    linear_examples = (loss_terms > 1)
    linear_loss = np.sum(loss_terms[linear_examples] - 0.5)
    return squared_loss + linear_loss


