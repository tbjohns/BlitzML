#include <blitzml/base/common.h>

#include <iostream>
#include <stdio.h>
#include <stdarg.h>

#ifdef BLITZML_R_WRAPPER
#include <Rcpp.h>
// https://cran.r-project.org/doc/manuals/r-release/R-exts.html#index-Rvprintf
#include <R.h>
#define vprintf Rvprintf
#define printf Rprintf
#endif

using std::string;
using std::cout;
using std::cerr;
using std::endl;

namespace BlitzML {

void assert_with_error_message(bool okay, string error_message) {
  if (!okay) {
#ifdef BLITZML_R_WRAPPER
    Rcpp::Rcerr << "Program exited with error: " << error_message << endl;
#else
    cerr << "Program exited with error: " << error_message << endl;
#endif
    throw error_message;
  }
}

void warn_if(bool condition, std::string message) {
  if (condition) {
#ifdef BLITZML_R_WRAPPER
    Rcpp::Rcout << "Warning: " << message << endl;
#else
    cout << "Warning: " << message << endl;
#endif
  }
}

void print(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  printf("\n");
  va_end(args);
}

void debug(const char* fmt, ...) {
#ifdef BLITZ_DEBUG
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  printf("\n");
  va_end(args);
#endif
}

} // namespace BlitzML

