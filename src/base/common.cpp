#include <blitzml/base/common.h>

#include <iostream>
#include <stdio.h>
#include <stdarg.h>

using std::string;
using std::cout;
using std::cerr;
using std::endl;


namespace BlitzML {

void assert_with_error_message(bool okay, string error_message) {
  if (!okay) {
    cerr << "Program exited with error: "
         << error_message
         << endl;
    throw error_message;
  }
}

void warn_if(bool condition, std::string message) {
  if (condition) {
    cout << "Warning: " << message << endl;
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

