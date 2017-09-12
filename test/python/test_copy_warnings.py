import unittest
import blitzml
import numpy as np
from scipy import sparse as sp

from common import captured_output

class TestCopyWarnings(unittest.TestCase):
  def setUp(self):
    self.A = np.ones((1000, 10001), dtype=np.float)
    self.b = np.ones(1000)

  def test_lasso_dense_copy_warning_C_contiguous(self):
    self.A = np.ascontiguousarray(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("Warning" in message)

  def test_lasso_dense_copy_warning_F_contiguous(self):
    self.A = np.asfortranarray(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("Warning" not in message)

  def test_lasso_csc_matrix_warning(self):
    self.A = sp.csc_matrix(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("Warning" not in message)

  def test_lasso_csr_matrix_warning(self):
    self.A = sp.csr_matrix(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("Warning" in message)


