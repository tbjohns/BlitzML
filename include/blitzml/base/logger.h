#pragma once

#include <blitzml/base/common.h>

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
namespace BlitzML {

class Logger {

  public:
    Logger();

    virtual ~Logger() { 
      if (main_log_file.is_open()) {
        main_log_file.close();
      }
    }

    void close_files();

    void set_log_directory(const char* directory);

    void throttle_logging_with_interval(double min_time_interval);

    bool log_new_point(value_t elapsed_time, value_t obj, bool force_no_throttle=false);

    void open_log_file(std::ofstream &log_file, const std::string &path) const;

    template <typename T>
    void log_value(std::string name, T value) const {
      if (throttle_point || !directory_is_valid || !main_log_file.is_open()) {
        return;
      }

      main_log_file << name << ": " << value << "\n";
    }

    template <typename T>
    void log_vector(std::string name, const std::vector<T> &vec) const {
      if (throttle_point || !directory_is_valid) {
        return;
      }

      std::ofstream log_file;
      open_log_file(log_file, get_filepath(name));
      if (log_file.fail()) {
        return;
      }

      for (size_t ind = 0; ind < vec.size(); ++ind) {
        log_file << vec[ind] << "\n";
      }

      log_file.close();
    }


  private:
    mutable std::ofstream main_log_file;
    bool throttle_point;
    std::string log_directory;
    double min_time_interval;
    double last_log_time;
    unsigned num_points_logged;
    bool directory_is_valid;

    const std::string get_filepath(const std::string &name) const;
    const std::string get_main_log_filepath() const;
    const std::string file_error_message(const std::string &filepath) const;
};

std::string unsigned2string(unsigned num);

} // namespace BlitzML
