import unittest
import blitzml
import numpy as np
from scipy import sparse as sp

from common import sparse_linear_problem_classes
from common import normalize_labels

class TestSparseLinearLambdaMax(unittest.TestCase):

  def setUp(self):
    blitzml.suppress_warnings()

  def tearDown(self):
    blitzml.unsuppress_warnings()

  def test_sparse_linear_lammax(self):
    A = np.arange(30).reshape(6, 5)
    b = np.arange(-2, 4)
    for problem_class in sparse_linear_problem_classes:
      prob = problem_class(A, b)
      lammax = prob.compute_max_l1_penalty()
      sol = prob.solve(lammax)
      sol2 = prob.solve(0.99 * lammax)
      self.assertAlmostEqual(np.linalg.norm(sol.weights), 0)
      self.assertNotEqual(np.linalg.norm(sol2.weights), 0)
      self.assertAlmostEqual(np.sum(sol.dual_solution), 0.)

  def test_sparse_linear_lammax_no_bias(self):
    A = np.arange(40).reshape(8, 5)
    b = np.arange(-2, 6)
    for problem_class in sparse_linear_problem_classes:
      prob = problem_class(A, b)
      lammax = prob.compute_max_l1_penalty(include_bias_term=False)
      sol = prob.solve(lammax, include_bias_term=False)
      sol2 = prob.solve(0.99 * lammax, include_bias_term=False)
      self.assertAlmostEqual(np.linalg.norm(sol.weights), 0)
      self.assertNotEqual(np.linalg.norm(sol2.weights), 0)

