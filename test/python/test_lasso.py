import unittest
import blitzml
import numpy as np
from scipy import sparse as sp

from common import matrix_vector_product


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


class TestLasso_1_by_1(unittest.TestCase):

  def setUp(self):
    A = np.array([[2]], dtype=np.float)
    b = np.array([10], dtype=np.double)
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
    b = np.zeros(100)
    prob = blitzml.LassoProblem(A, b)
    self.assertEqual(prob.compute_max_l1_penalty(), 0.)
    sol = prob.solve(0.)
    self.assertEqual(np.linalg.norm(sol.weights), 0.)
    self.assertEqual(sol.bias, 0.)
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



