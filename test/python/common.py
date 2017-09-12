import numpy as np
import sys
from contextlib import contextmanager
try:
  # Python2
  from StringIO import StringIO
except ImportError:
  # Python3
  from io import StringIO

def matrix_vector_product(A, v):
  return np.array(A * np.mat(v).T).flatten()

@contextmanager
def captured_output():
  new_out = StringIO()
  old_out = sys.stdout
  try:
    sys.stdout = new_out
    yield sys.stdout
  finally:
    sys.stdout = old_out

