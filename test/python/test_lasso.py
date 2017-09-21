import unittest
import blitzml
import numpy as np
from scipy import sparse as sp

from common import matrix_vector_product
from common import captured_output


def is_solution(sol, A, b, lam, tol=1e-3):
  duals = sol.bias + matrix_vector_product(A, sol.weights) - b
  grads = matrix_vector_product(A.T, duals)
  max_grads = np.max(abs(grads))
  if max_grads > lam * (1 + tol):
    return False
  pos_grads_diff = grads[sol.weights > 0] + lam
  if len(pos_grads_diff) and max(abs(pos_grads_diff)) > lam * tol:
    return False
  neg_grads_diff = grads[sol.weights < 0] - lam
  if len(neg_grads_diff) and max(abs(neg_grads_diff)) > lam * tol:
    return False
  return True


def compute_dual_objective(theta, b):
  return -0.5 * np.linalg.norm(theta) ** 2 - np.dot(theta, b)


class TestLasso_1_by_1(unittest.TestCase):

  def setUp(self):
    A = np.array([[2]], dtype=np.float)
    b = np.array([10], dtype=np.double)
    with captured_output() as out:
      self.prob = blitzml.LassoProblem(A, b)

  def test_lasso_1_by_1(self):
    sol = self.prob.solve(5., include_bias_term=False)
    self.assertAlmostEqual(sol.weights[0], 15./4)
    self.assertEqual(sol.bias, 0.)
    self.assertAlmostEqual(sol.dual_solution[0], -2.5)
    self.assertIn(sol.solution_status, {"Reached stopping tolerance", "Reached machine precision"})
    self.assertAlmostEqual(sol.duality_gap, 0.)
    self.assertAlmostEqual(sol.objective_value, 5*15./4 + 0.5 * (2.5)**2)

  def test_lasso_1_by_1_with_bias(self):
    sol = self.prob.solve(5., include_bias_term=True)
    self.assertEqual(sol.weights[0], 0.)
    self.assertEqual(sol.bias, 10.)
    self.assertEqual(sol.dual_solution[0], 0.0)
    self.assertIn(sol.solution_status, {"Reached stopping tolerance", "Reached machine precision"})
    self.assertEqual(sol.duality_gap, 0.)
    self.assertEqual(sol.objective_value, 0.)


class TestLassoEmpty(unittest.TestCase):
  def test_lasso_empty(self): 
    A = sp.csc_matrix((100, 1000))
    b = np.ones(100)
    with captured_output() as out:
      prob = blitzml.LassoProblem(A, b)
    self.assertEqual(prob.compute_max_l1_penalty(), 0.)
    sol = prob.solve(0.)
    self.assertEqual(np.linalg.norm(sol.weights), 0.)
    self.assertEqual(sol.bias, 1.0)
    self.assertEqual(sol.objective_value, 0.)
    self.assertEqual(sol.duality_gap, 0.)

  def test_lasso_empty_cols(self):
    np.random.seed(1)
    A = np.random.randn(20, 30)
    A[:,0] = 0. 
    A[:, 10] = 0.  
    A[:, 29] = 0.
    A_sparse = sp.csc_matrix(A)
    b = np.random.randn(20) 
    prob = blitzml.LassoProblem(A_sparse, b)
    lammax = prob.compute_max_l1_penalty()
    lam = 0.05 * lammax
    sol = prob.solve(lam, stopping_tolerance=1e-5)
    self.assertEqual(is_solution(sol, A, b, lam), True)


class TestLassoInitialConditions(unittest.TestCase):
  def test_lasso_bad_initial_conditions(self):
    n = 6
    d = 4
    A = np.arange(n * d).reshape(n, d)
    b = np.arange(n)
    prob = blitzml.LassoProblem(A, b)
    lammax = prob.compute_max_l1_penalty()
    weights0 = -100 * np.arange(d)
    lam = 0.02 * lammax
    sol = prob.solve(lam, initial_weights=weights0, stopping_tolerance=1e-5)
    self.assertEqual(is_solution(sol, A, b, lam), True)

  def test_lasso_good_initial_conditions(self):
    n = 7
    d = 30
    np.random.seed(0)
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    prob = blitzml.LassoProblem(A, b)
    lammax = prob.compute_max_l1_penalty()
    lam = 0.005 * lammax
    sol0 = prob.solve(lam, stopping_tolerance=1e-5)
    sol = prob.solve(lam, initial_weights=sol0.weights, max_time=-1.0)
    self.assertEqual(is_solution(sol, A, b, lam), True)


class TestLassoObjectives(unittest.TestCase):
  def setUp(self):
    np.random.seed(1) 
    self.A = np.random.uniform(0, 1, size=(20, 10))
    self.b = np.random.uniform(0, 1, size=20)
    prob = blitzml.LassoProblem(self.A, self.b)
    self.lam = 0.02 * prob.compute_max_l1_penalty()
    self.sol = prob.solve(self.lam) 

  def test_lasso_compute_loss(self):
    d = self.A.shape[1]
    A = np.zeros((0, d))
    b = np.zeros(0)
    self.assertEqual(self.sol.compute_loss(A, b), 0.)

    A = np.random.randn(1, d)
    b = np.random.randn(1, 1)
    Aw = np.dot(A[0,:], self.sol.weights) + self.sol.bias
    loss = 0.5 * (Aw - b[0]) ** 2
    self.assertEqual(self.sol.compute_loss(A, b), loss)

  def test_lasso_primal_objective(self):
    l1_norm = np.linalg.norm(self.sol.weights, ord=1)
    obj = self.sol.compute_loss(self.A, self.b) + self.lam * l1_norm
    self.assertAlmostEqual(obj, self.sol.objective_value)

  def test_lasso_dual_solution(self):
    dual_obj = compute_dual_objective(self.sol.dual_solution, self.b)
    sol_dual_obj = self.sol.objective_value - self.sol.duality_gap
    self.assertAlmostEqual(dual_obj, sol_dual_obj)  

  def test_lasso_predict(self):
    d = self.A.shape[1]
    A = np.random.randn(2, d)
    Aw = np.dot(A, self.sol.weights) + self.sol.bias
    diff = np.linalg.norm(Aw - self.sol.predict(A))
    self.assertEqual(diff, 0.)  
    
class TestLassoLambdaMax(unittest.TestCase):
  def test_sparse_linear_lambda_max_basic(self):
    A = np.array([[1, -2], [1, 0]]).T
    b = np.array([3, -1])
    prob = blitzml.LassoProblem(A, b)
    lammax = prob.compute_max_l1_penalty()
    self.assertEqual(lammax, 6.0)
    lammax2 = prob.compute_max_l1_penalty(include_bias_term=False)
    self.assertEqual(lammax2, 5.0)








