#include <blitzml/base/timer.h>
#include <iostream>

#include <time.h>

namespace BlitzML {

#ifdef WINDOWS_OS

#include "Windows.h"

double get_time() {
  LARGE_INTEGER count;
  QueryPerformanceCounter(&count);
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  double t = static_cast<double>(count.QuadPart) / freq.QuadPart;
  debug("Time is %0.15f\n", t);
  return t;
}

#else // no Windows
#ifdef CLOCK_MONOTONIC

double get_time() {
  timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_nsec) / 1e9;
}

#else // no CLOCK_MONOTONIC

#include <sys/time.h>
#include <sys/resource.h>

double timeval_to_seconds(const timeval &time) {
  return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec) / 1e6;
}

double get_time() {
  rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return timeval_to_seconds(usage.ru_utime) + timeval_to_seconds(usage.ru_stime);
}

#endif // no CLOCK_MONTONIC
#endif // no Windows


Timer::Timer() {
  reset();
}


double Timer::current_time() {
  double t = get_time();
  if (t <= last_get_time) {
    t = last_get_time;
  }
  last_get_time = t;
  return t;
}


void Timer::reset() {
  pause_accum = 0.;
  absolute_start_time = current_time();
  is_paused = false;
}


double Timer::elapsed_time() {
  if (is_paused) {
    return absolute_pause_time - absolute_start_time - pause_accum;
  } else {
    return current_time() - absolute_start_time - pause_accum;
  }
}


void Timer::pause_timing() {
  absolute_pause_time = current_time();
  is_paused = true;
}


void Timer::continue_timing() {
  pause_accum += current_time() - absolute_pause_time;
  is_paused = false;
}


} // namespace BlitzML
