import unittest
from contextlib import contextmanager
try:
  # Python2
  from StringIO import StringIO
except ImportError:
  # Python3
  from io import StringIO
import sys
import blitzml
import numpy as np
from scipy import sparse as sp


@contextmanager
def captured_output():
  new_out = StringIO()
  old_out = sys.stdout
  try:
    sys.stdout = new_out
    yield sys.stdout
  finally:
    sys.stdout = old_out

#@unittest.skip("Takes a bit of time to run")
class TestCopyWarnings(unittest.TestCase):
  def setUp(self):
    self.A = np.ones((1000, 10001), dtype=np.float)
    self.b = np.ones(1000)

  def test_lasso_dense_copy_warning_C_contiguous(self):
    self.A = np.ascontiguousarray(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("warning" in message)

  def test_lasso_dense_copy_warning_F_contiguous(self):
    self.A = np.asfortranarray(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("warning" not in message)

  def test_lasso_csc_matrix_warning(self):
    self.A = sp.csc_matrix(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("warning" not in message)

  def test_lasso_csr_matrix_warning(self):
    self.A = sp.csr_matrix(self.A)
    with captured_output() as out:
      prob = blitzml.LassoProblem(self.A, self.b)
    message = out.getvalue()
    self.assertTrue("warning" in message)


