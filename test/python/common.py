import blitzml
import numpy as np
import sys
import os
from contextlib import contextmanager

def matrix_vector_product(A, v):
  return np.array(A * np.mat(v).T).flatten()

@contextmanager
def captured_output(to="tmp_out_file"):
  fd = sys.stdout.fileno()
  result = []

  def redirect_stdout(to):
    sys.stdout.close()
    os.dup2(to.fileno(), fd)
    sys.stdout = os.fdopen(fd, 'w')

  with os.fdopen(os.dup(fd), 'w') as old_stdout:
    with open(to, 'w') as f:
      redirect_stdout(to=f)
    try:
      yield result
    finally:
      redirect_stdout(to=old_stdout)
      with open(to) as out_f:
        result.append(out_f.read())
      os.remove(to)

sparse_linear_problem_classes = [
  blitzml.LassoProblem,
  blitzml.SparseLogisticRegressionProblem,
  blitzml.SparseHuberProblem,
  blitzml.SparseSquaredHingeProblem,
  blitzml.SparseSmoothedHingeProblem
]

def normalize_labels(b, plus_minus_one_only=False):
    b = b.astype(np.float)
    b -= min(b)
    b /= max(b)
    b = 2 * b - 1.0
    if plus_minus_one_only:
      b[b <= 0] = -1.0
      b[b > 0] = 1.0
    return b


