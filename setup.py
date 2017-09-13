from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os
import sys

setup_dir = "."
library_dir = "."
package_dir = "./python-package"

sys.path.append(package_dir)
from _lib_file_extension import get_lib_file_extension

def source_files_generator():
  for path, subdirs, files in os.walk(os.path.join(library_dir, "src")):
    for name in files:
      if name.endswith("cpp"):
        yield os.path.join(path, name)

ext = Extension("blitzml.libblitzml",
                sources=list(source_files_generator()),
                extra_compile_args = ["-O3"], 
                extra_link_args = [],
                include_dirs=[os.path.join(library_dir, "include")]
) 

def new_get_export_symbols(self, ext):
  return ext.export_symbols
build_ext.get_export_symbols = new_get_export_symbols

def new_get_ext_filename(self, ext_name):
  return "{}.{}".format(ext_name, get_lib_file_extension())
build_ext.get_ext_filename = new_get_ext_filename

version_path = os.path.join(library_dir, "VERSION")
with open(version_path) as versionf:
  version_string = versionf.read().strip()

setup(name = "blitzml",
      version=version_string,
      description="BlitzML optimization library for machine learning",
      package_dir={"blitzml" : os.path.join(setup_dir, "python-package")},
      packages=["blitzml"],
      ext_modules=[ext],
      author="Tyler Johnson",
      author_email="tyler@tbjohns.com",
      license="BSD-3-Clause",
      classifiers=["License :: OSI Approved :: BSD License"],
      cmdclass={"buildext": build_ext}
     )

