#pragma once

#include <vector>
#include <string>
#include <limits>
#include <iostream>

namespace BlitzML {

#if defined(_WIN32) || defined(_WIN64)
#define WINDOWS_OS 1
#endif

#ifdef WINDOWS_OS
#define LIBRARY_API extern "C" __declspec(dllexport)
#else
#define LIBRARY_API extern "C"
#endif

typedef double value_t;
//typedef std::size_t size_t;
typedef uint32_t size_t;
typedef uint32_t index_t;
//typedef int index_t;
typedef std::vector<index_t>::iterator index_itr;
typedef std::vector<value_t>::iterator value_itr;
typedef std::vector<index_t>::const_iterator const_index_itr;
typedef std::vector<value_t>::const_iterator const_value_itr;


void assert_with_error_message(bool okay, std::string error_message);
void warn_if(bool condition, std::string message);
void print(const char* fmt, ...);
void debug(const char* fmt, ...);


class ObjectiveValues {
  public:
    ObjectiveValues()
      : dual_obj_(std::numeric_limits<value_t>::min()),
        primal_obj_x_(std::numeric_limits<value_t>::max()),
        primal_obj_y_(std::numeric_limits<value_t>::max()),
        primal_obj_z_(std::numeric_limits<value_t>::max()) { }

    virtual ~ObjectiveValues() { }

    value_t duality_gap() const { return primal_obj_y_ - dual_obj_; }

    void set_primal_obj_x(value_t val) { primal_obj_x_ = val; }
    value_t primal_obj_x() const { return primal_obj_x_; }

    void set_primal_obj_y(value_t val) { primal_obj_y_ = val; }
    value_t primal_obj_y() const { return primal_obj_y_; }

    void set_primal_obj_z(value_t val) { primal_obj_z_ = val; }
    value_t primal_obj_z() const { return primal_obj_z_; }

    void set_dual_obj(value_t val) { dual_obj_ = val; }
    value_t dual_obj() const { return dual_obj_; }

  private:
    value_t dual_obj_;
    value_t primal_obj_x_;
    value_t primal_obj_y_;
    value_t primal_obj_z_;
};


struct SubproblemState {
  std::vector<value_t> x;
  value_t dual_obj;

  virtual ~SubproblemState() { }
};


class Parameters {
  public:
    Parameters(const value_t *values, size_t num_parameters)
        : values(values), count(num_parameters) { }

    size_t size() const { return count; }
    value_t operator[] (size_t i) const { return values[i]; }

    virtual ~Parameters() { debug("delete params"); }

  private:
    const value_t *values;
    size_t count;

    Parameters();
};


struct SubproblemParameters {
  double time_limit;
  double min_time;
  value_t epsilon;
  value_t xi;
  int max_iterations;

  virtual ~SubproblemParameters() { }
};

} // namespace BlitzML

