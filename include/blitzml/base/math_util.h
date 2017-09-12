#pragma once

#include <blitzml/base/common.h>

#include <algorithm>

#include <vector>
#include <cmath>
#include <numeric>

namespace BlitzML {


template <typename T>
inline T sq(T value) {
  return value * value;
}


template <typename T>
inline T cube(T value) {
  return value * value * value;
}


template <typename T>
inline T sign(T value) {
  return (value < 0) ? -1 : 1;
}

template <typename T>
struct AccumSq {
  T operator() (T result, T val) {
    return result + val * val;
  }
};

template <typename T>
inline value_t l2_norm_sq(const T* values, size_t length) {
  value_t result =  0.0;
  for (size_t ind = 0; ind < length; ++ind) {
    result += sq(values[ind]);
  }
  return result;
}


template <typename T>
inline value_t l2_norm_sq(const std::vector<T> &vec) {
  return l2_norm_sq(&vec[0], vec.size());
}


template <typename T>
inline value_t l2_norm_diff_sq(const std::vector<T> &vec1, 
                               const std::vector<T> &vec2) {
  value_t result = 0.;
  for (size_t ind = 0; ind < vec1.size(); ++ind) {
    result += sq(vec1[ind] - vec2[ind]);
  }
  return result;
}


template <typename T>
inline value_t l2_norm_diff(const std::vector<T> &vec1, 
                            const std::vector<T> &vec2) {
  return std::sqrt(l2_norm_diff_sq(vec1, vec2));
}


template <typename T>
inline value_t l1_norm(const T* values, size_t len) {
  value_t result = 0.0;
  for (size_t ind = 0; ind < len; ++ind) {
    if (values[ind] != 0.0) {
      result += fabs(values[ind]);
    }
  }
  return result;
}


template <typename T>
inline value_t l1_norm(const std::vector<T> &vec) {
  return l1_norm(&vec[0], vec.size());
}


template <typename T>
inline value_t sum_array(const T* values, size_t length) {
  value_t result =  0.;
  for (size_t ind = 0; ind < length; ++ind) {
    result += values[ind];
  }
  return result;
}


template <typename T>
inline value_t sum_vector(const std::vector<T> &vec) {
  return sum_array(&vec[0], vec.size());
}


template <typename T>
inline size_t l0_norm(const std::vector<T> &vec) {
  size_t result = 0;
  for (size_t ind = 0; ind < vec.size(); ++ind) {
    if (vec[ind] != 0) {
      ++result;
    }
  }
  return result;
}


template <typename T>
inline size_t l0_norm(const T* values, size_t len) {
  size_t result = 0;
  for (size_t ind = 0; ind < len; ++ind) {
    if (values[ind] != 0) {
      ++result;
    }
  }
  return result;
}


template <typename T>
inline value_t inner_product(const T* values1, const T* values2, size_t size) {
  return std::inner_product(values1, values1 + size, values2, 0.);
}


template <typename T>
inline value_t inner_product(const std::vector<T> &vec1, 
                             const std::vector<T> &vec2) {
  return inner_product(&vec1[0], &vec2[0], vec1.size());
}


template <typename T>
inline value_t soft_threshold(T value, T threshold) {
  if (value > threshold) {
    return value - threshold;
  }
  if (value < -threshold) {
    return value + threshold;
  }
  return 0.;
}


template <typename T>
inline T max_abs(const std::vector<T> &vec) {
  T result = 0;
  for (size_t ind = 0; ind < vec.size(); ++ind) {
    if (fabs(vec[ind]) > result)
      result = fabs(vec[ind]);
  }
  return result;
}


template <typename T>
inline T max_vector(const std::vector<T> & vec) {
  if (vec.size() == 0) {
    return 0;
  }
  return *std::max_element(vec.begin(), vec.end());
}


template <typename T>
T median(const std::vector<T>& values, size_t first, size_t last) {
  std::vector<T> values_of_interest(values.begin() + first, 
                                    values.begin() + last);
  size_t mid = values_of_interest.size() / 2;
  std::sort(values_of_interest.begin(), values_of_interest.end());
  if (first == last) {
    return 0.;
  } else if ((last - first) % 2 == 1) {
    return values_of_interest[mid];
  }
  return (values_of_interest[mid] + values_of_interest[mid - 1]) / 2;
}


template <typename T>
T median_last_k(const std::vector<T>& values, size_t k) {
  if (values.size() >= k) {
    return median(values, values.size() - k, values.size());
  } else {
    return median(values, 0, values.size());
  }
}


std::pair<value_t, value_t> compute_quadratic_roots(
                                value_t a, value_t b, value_t c);


} // namespace BlitzML

