import unittest
import blitzml
import numpy as np

from common import captured_output

class TestProblemOptions(unittest.TestCase):
  def setUp(self):
    A = np.arange(20).reshape(5, 4)
    b = np.arange(5).astype(np.float64)
    self.prob = blitzml.LassoProblem(A, b)

  def tearDown(self):
    del self.prob

  def test_min_time(self):
    self.assertLessEqual(self.prob._min_time, 0.)
    self.prob._min_time = 2.0
    self.assertEqual(self.prob._min_time, 2.0)

  def test_max_time(self):
    self.assertGreaterEqual(self.prob._max_time, 3600.)
    self.prob._max_time = 5.0
    self.assertEqual(self.prob._max_time, 5.0)

  def test_max_iterations(self):
    self.assertGreaterEqual(self.prob._max_iterations, 100)
    self.prob._max_iterations = 10
    self.assertEqual(self.prob._max_iterations, 10)

  def test_tolerance(self):
    self.assertGreater(self.prob._stopping_tolerance, 0.)
    self.prob._stopping_tolerance = 0.
    self.assertEqual(self.prob._stopping_tolerance, 0.)
    self.prob._stopping_tolerance = 0.1
    self.assertEqual(self.prob._stopping_tolerance, 0.1)

  def test_verbose(self):
    self.assertEqual(self.prob._verbose, False)
    self.prob._verbose = True
    self.assertEqual(self.prob._verbose, True)

  def test_use_screening(self):
    self.assertEqual(self.prob._use_screening, True)
    self.prob._use_screening = False
    self.assertEqual(self.prob._use_screening, False)

  def test_use_working_sets(self):
    self.assertEqual(self.prob._use_working_sets, True)
    self.prob._use_working_sets = False
    self.assertEqual(self.prob._use_working_sets, False)

  def test_suppress_warnings(self):
    bad_log_dir = "path/to/bad_log/dir/zxc8aj3n"
    with captured_output() as out:
      self.prob.solve(self.prob.compute_max_l1_penalty(),
                      log_directory=bad_log_dir)
    self.assertIn("Warning", out[0])

    blitzml.suppress_warnings()

    with captured_output() as out:
      self.prob.solve(self.prob.compute_max_l1_penalty(),
                      log_directory=bad_log_dir)
    self.assertNotIn("Warning", out[0])

    blitzml.unsuppress_warnings()

