#include <blitzml/base/logger.h>

#include <limits>

using std::string;
using std::ofstream;
using std::ostringstream;

namespace BlitzML {

Logger::Logger() :
      log_directory(""),
      min_time_interval(0.),
      last_log_time(0.),
      num_points_logged(0) { } 


void Logger::close_files() {
  if (main_log_file.is_open()) {
    main_log_file.close();
  }
}


void Logger::set_log_directory(const char* directory) { 
  log_directory = directory; 
  directory_is_valid = (log_directory.length() > 0) ? true : false;
  if (!directory_is_valid) {
    return;
  }
  open_log_file(main_log_file, get_main_log_filepath());
  if (main_log_file.fail()) {
    directory_is_valid = false;
  }
}


void Logger::throttle_logging_with_interval(double min_time_interval) {
  this->min_time_interval = min_time_interval;
}


bool Logger::log_new_point(double elapsed_time, value_t obj, bool force_no_throttle) {
  if (!directory_is_valid) {
    return false;
  }

  throttle_point = (num_points_logged > 0) && 
                   (elapsed_time < last_log_time + min_time_interval) &&
                   (!force_no_throttle);
  if (throttle_point) {
    return false;
  }

  ++num_points_logged;

  log_value<unsigned>("log_point_number", num_points_logged);
  log_value<double>("time", elapsed_time);
  log_value<value_t>("dual_obj", obj);

  last_log_time = elapsed_time;
  return true;
}


void Logger::open_log_file(ofstream &log_file, const string &path) const {
  log_file.open(path.c_str());
  log_file.precision(15);
  warn_if(log_file.fail(), file_error_message(path));
}


const string Logger::get_filepath(const string &name) const {
  return log_directory + name + "." + unsigned2string(num_points_logged) + ".log";
}


const string Logger::get_main_log_filepath() const {
  return log_directory + "main.log";
}


const string Logger::file_error_message(const string &filepath) const {
  return "could not open file " + filepath + " for logging";
}


string unsigned2string(unsigned num) {
  ostringstream ss;
  ss << num;
  return ss.str();
}


} // namespace BlitzML
