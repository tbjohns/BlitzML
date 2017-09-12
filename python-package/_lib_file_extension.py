# coding: utf-8

import os

def get_lib_file_extension():
  if os.name == 'nt':
    # Windows
    return "dll"
  else:
    return "so"

