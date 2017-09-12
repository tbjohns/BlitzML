#pragma once

#include "common.h"

namespace BlitzML {

double get_time();

class Timer {
  public:
    Timer();
    virtual ~Timer() { } 
    double current_time();
    void reset();
    double elapsed_time();
    void pause_timing();
    void continue_timing();

  private:
    double absolute_start_time;
    double absolute_pause_time;
    double pause_accum;
    double last_get_time;
    bool is_paused;
};

} // namespace BlitzML

