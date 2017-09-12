#pragma once

#include <blitzml/base/common.h>

#include <algorithm>
#include <utility>
#include <cmath>

namespace BlitzML {

template <typename T>
void crude_shuffle_helper(std::vector<T> &vec, size_t first, size_t last, size_t shift) {
  if (last <= first) {
    return;
  }
  size_t diff = last - first;
  size_t shift_accum = 0;
  for (size_t ind1 = first; ind1 != last; ++ind1) {
    shift_accum += shift;
    shift_accum %= diff;
    std::swap(vec[ind1], vec[first + shift_accum]);
  }
}


template <typename T>
void crude_shuffle(std::vector<T> &vec, size_t first, size_t last, int seed=0) {
  size_t shift1, shift2; 
  switch (seed % 5) {
    case 0:
      shift1 = 59;
      shift2 = 97;
      break;
    case 1:
      shift1 = 13;
      shift2 = 149;
      break;
    case 2:
      shift1 = 7;
      shift2 = 67;
      break;
    case 3:
      shift1 = 251;
      shift2 = 17;
      break;
    default:
      shift1 = 191;
      shift2 = 11;
      break;
  }
  crude_shuffle_helper(vec, first, last, shift1);
  crude_shuffle_helper(vec, first, last, shift2);
}


class IndirectComparator {
  public:
    IndirectComparator(const std::vector<value_t> &v) : values(v) {}
    inline bool operator() (const index_t &i, const index_t &j) {
      return values[i] < values[j];
    }

    virtual ~IndirectComparator() { }

  private:
    const std::vector<value_t>& values;
};


inline void indirect_sort_indices(std::vector<index_t> &indices, 
                                  const std::vector<value_t> &values_lookup) {
  IndirectComparator cmp(values_lookup);
  std::sort(indices.begin(), indices.end(), cmp);
}


template <typename T>
inline void scale_vector(std::vector<T> &vec, register T scale) {
  for (size_t ind = 0; ind < vec.size(); ++ind) {
    vec[ind] *= scale;
  }
}


template <typename T>
inline void add_scalar_to_vector(std::vector<T> &vec, register T scalar) {
  for (size_t ind = 0; ind < vec.size(); ++ind) {
    vec[ind] += scalar;
  }
}


template <typename T>
inline bool is_vector_const(const std::vector<T> &vec, T tolerance=0) {
  if (vec.size() == 0) {
    return true;
  }
  T val = vec[0];
  for (size_t ind = 0; ind < vec.size(); ++ind) {
    if (std::fabs(vec[ind] - val) > tolerance) {
      return false;
    }
  }
  return true;
}


} // namespace BlitzML

