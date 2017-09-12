import unittest
import blitzml
import numpy as np
from scipy import sparse as sp


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

