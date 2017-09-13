# coding: utf-8

from blitzml._core import warn

import os as os
import numpy as np
try:
  import pandas as pd
  have_pandas = True
except ImportError:
  have_pandas = False


def parse_log_directory(log_directory):
  """Parse files logged by BlitzML during a solve call.

  Parameters
  ----------
  log_directory : string
    Path to directory containing log files to parse.

  Returns
  -------
  logs : generator
    Iterable over information logged by BlitzML. Each item is a dictionary of
    logged values.
  """
  parser = _LogParser(log_directory)
  return parser.get_log_points()


def lines_form_file(path):
  with open(path) as f:
    for line in f:
      yield line


def format_value(str_value):
  try:
    return int(str_value)
  except ValueError:
    pass
  try:
    return float(str_value)
  except ValueError:
    pass
  return str_value.strip()


def load_list_from_file(path):
  is_empty = ( os.path.getsize(path) == 0 )
  if is_empty:
    return np.array([])
  dtype = None
  with open(path) as f:
    first_line = f.readline()
    dtype = type(format_value(first_line))
  if have_pandas:
    return load_list_from_file_pandas(path, dtype)
  else:
    return load_list_from_file_numpy(path, dtype)


def load_list_from_file_pandas(path, dtype):
  return np.array(pd.read_fwf(path), dtype=dtype).squeeze()


def load_list_from_file_numpy(path, dtype):
  return np.loadtxt(path, dtype=dtype)


class _LogParser(object):
  def __init__(self, log_directory):
    self._dir = log_directory
    self._set_valid_dir()
    self._set_list_names()

  @property
  def _main_filepath(self):
    return os.path.join(self._dir, "main.log")
    
  def _set_valid_dir(self):
    self._valid_dir = True
    if not os.path.exists(self._dir):
      self._valid_dir = False
      warn("Log directory not found.")
    if not os.path.exists(self._main_filepath):
      self._valid_dir = False
      warn("No main.log found in provided log directory.")

  def _set_list_names(self):
    filenames_1 = (n for n in os.listdir(self._dir) if n.endswith(".1.log"))
    self._list_names = [n[:-6] for n in filenames_1]

  def get_log_points(self):
    if not self._valid_dir:
      return
    d = {}
    log_point_number = 1
    for line in lines_form_file(self._main_filepath):
      split = line.split(":")
      key = split[0]
      value = format_value(split[1])
      if key == "log_point_number" and d:
        yield self._add_vectors(d, log_point_number)
        d = {}
        log_point_number += 1
      d[key] = value
    yield self._add_vectors(d, log_point_number)

  def _add_vectors(self, d, log_point_number):
    for list_name in self._list_names:
      filename = "{}.{:d}.log".format(list_name, log_point_number)
      path = os.path.join(self._dir, filename)
      d[list_name] = load_list_from_file(path)
    return d
    

