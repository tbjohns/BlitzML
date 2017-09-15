import unittest
import blitzml
import numpy as np
from scipy import sparse as sp

from common import captured_output
from common import matrix_vector_product
from common import normalize_labels

def is_solution(sol, A, b, lam, tol=1e-3):
  Aomega = sol.bias + matrix_vector_product(A, sol.weights) 
  exp_bAomega = np.exp(b * Aomega)
  grads = matrix_vector_product(A.T, -b / (1 + exp_bAomega))
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



class TestSparseLogRegInitialConditions(unittest.TestCase):
  def test_sparse_logreg_bad_initial_conditions(self):
    n = 7
    d = 3
    A = np.arange(n * d).reshape(n, d)
    b = normalize_labels(np.arange(n), True)
    prob = blitzml.SparseLogisticRegressionProblem(A, b)
    lammax = prob.compute_max_l1_penalty()
    weights0 = -1 * np.arange(d)
    lam = 0.02 * lammax
    sol = prob.solve(lam, initial_weights=weights0, stopping_tolerance=1e-6)
    self.assertEqual(is_solution(sol, A, b, lam), True)

  def test_sparse_logreg_good_initial_conditions(self):
    n = 9
    d = 21
    np.random.seed(0)
    A = np.random.randn(n, d)
    b = normalize_labels(np.random.randn(n), True)
    prob = blitzml.SparseLogisticRegressionProblem(A, b)
    lammax = prob.compute_max_l1_penalty()
    lam = 0.03 * lammax
    sol0 = prob.solve(lam, stopping_tolerance=1e-4, max_time=1.0)
    sol = prob.solve(lam, initial_weights=sol0.weights, max_time=-1.0)
    self.assertEqual(is_solution(sol, A, b, lam), True)


class TestSparseLogRegBadLabels(unittest.TestCase):
  def test_sparse_logreg_bad_label_too_large(self):
    b = np.array([-1., 0., 2.])
    A = np.zeros((3, 3))
    with captured_output() as out:
      prob = blitzml.SparseLogisticRegressionProblem(A, b)
    message = out[0]
    self.assertIn("Warning", message)

  def test_sparse_logreg_bad_label_too_small(self):
    b = np.array([-1., 0., -2.])
    A = np.zeros((3, 3))
    with captured_output() as out:
      prob = blitzml.SparseLogisticRegressionProblem(A, b)
    message = out[0]
    self.assertIn("Warning", message)

  def test_sparse_logreg_dimension_mismatch(self):
    b = np.array([-1., 0., -2.])
    A = np.zeros((2, 3))
    def make_prob():
      prob = blitzml.SparseLogisticRegressionProblem(A, b)
    self.assertRaises(ValueError, make_prob)

  def test_sparse_logreg_all_positive_labels_warning(self):
    b = np.array([0., 1.0, 0.5])
    A = np.zeros((3, 3))
    with captured_output() as out:
      prob = blitzml.SparseLogisticRegressionProblem(A, b)
    message = out[0]
    self.assertIn("Warning", message)


