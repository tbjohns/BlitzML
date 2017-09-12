#include <blitzml/base/working_set.h>

namespace BlitzML {

void WorkingSet::clear() { 
  size_working_set = 0; 
  in_working_set.assign(max_size, false);
}

void WorkingSet::set_max_size(size_t size) {
  working_set.resize(size);
  sorted_working_set.resize(size);
  indices.resize(size);
  in_working_set.resize(size);
  max_size = size;
}

void WorkingSet::reduce_max_size(size_t new_size) { 
  max_size = new_size; 
}

void WorkingSet::shuffle() {
  ++shuffle_count;
  crude_shuffle(working_set, 0, size_working_set, shuffle_count);
}

void WorkingSet::shuffle_indices() {
  ++shuffle_count;
  crude_shuffle(indices, 0, size_working_set, shuffle_count);
}

}

