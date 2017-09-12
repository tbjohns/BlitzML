#pragma once 

#include <blitzml/base/common.h>
#include <blitzml/base/vector_util.h>

#include <vector>

namespace BlitzML {

class WorkingSet {

  public:
    WorkingSet() : shuffle_count(0) { }
    virtual ~WorkingSet() { }

    void clear();
    void set_max_size(size_t size);
    void reduce_max_size(size_t new_size);

    inline size_t size() const {
      return size_working_set; 
    }

    inline void add_index(index_t i) {
      working_set[size_working_set] = i;
      sorted_working_set[size_working_set] = i;
      indices[size_working_set] = static_cast<index_t>(size_working_set);
      in_working_set[i] = true;
      ++size_working_set;
    }

    inline index_t ith_member(index_t i) const {
      return working_set[i];
    }

    inline bool is_in_working_set(index_t i) const {
      return in_working_set[i];
    }

    void shuffle(); 
    void shuffle_indices();

    inline const_index_itr begin() const { 
      return working_set.begin(); 
    }

    inline const_index_itr end() const { 
      return working_set.begin() + size_working_set; 
    }

    inline const_index_itr begin_sorted() const { 
      return sorted_working_set.begin(); 
    }

    inline const_index_itr end_sorted() const { 
      return sorted_working_set.begin() + size_working_set; 
    }

    inline const_index_itr begin_indices() const { 
      return indices.begin(); 
    }

    inline const_index_itr end_indices() const { 
      return indices.begin() + size_working_set; 
    }

  private:
    std::vector<index_t> working_set;
    std::vector<index_t> indices;
    std::vector<index_t> sorted_working_set;
    std::vector<bool> in_working_set;
    size_t size_working_set;
    size_t max_size;
    int shuffle_count;
};

}
