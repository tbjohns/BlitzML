#pragma once

#include <blitzml/base/common.h>

namespace BlitzML {

class Column {
  public:
    Column() { }
    Column(index_t length, index_t nnz) : length_(length), nnz_(nnz) { }

    virtual ~Column() { }

    index_t length() const { return length_; }
    index_t nnz() const { return nnz_; } 

    virtual value_t inner_product(const std::vector<value_t> &vec) const = 0;
    virtual value_t weighted_inner_product(
        const std::vector<value_t> &vec, 
        const std::vector<value_t> &weights) const = 0;
    virtual value_t weighted_norm_sq(
        const std::vector<value_t> &weights) const = 0;

    virtual void add_multiple(
        std::vector<value_t> &target, value_t scalar) const = 0;
    virtual void weighted_add_multiple(
        std::vector<value_t> &target, std::vector<value_t> &weights, value_t scalar) const = 0;

    virtual value_t sum() const = 0;
    virtual value_t mean() const = 0;

    virtual value_t l2_norm_sq() const = 0;
    value_t l2_norm() const;
    value_t l2_norm_sq_centered() const;
    value_t l2_norm_centered() const;

  protected:
    index_t length_;
    index_t nnz_;
};


class Dataset {
  public:
    Dataset(index_t height, index_t width, size_t nnz, 
            const value_t *b, index_t length_b) 
        : height(height), width(width), nnz_(nnz), b(b), length_b(length_b) { }

    virtual ~Dataset() { debug("delete dataset"); }

    virtual const Column* column(index_t j) const = 0;
    index_t num_rows() const { return height; }
    index_t num_cols() const { return width; }
    size_t nnz() const { return nnz_; }

    inline value_t b_value(index_t i) const { return b[i]; }
    const value_t* b_values() const { return b; }

    virtual void contiguous_submatrix_multiply(
        const std::vector<value_t> &values, value_t* result, 
        index_t first_column_submatrix, index_t end_column_submatrix) const;
    
    virtual bool any_columns_overlapping_in_submatrix(
        index_t first_column_submatrix, 
        index_t end_column_submatrix) const = 0;

  protected:
    index_t height; 
    index_t width; 
    size_t nnz_; 
    const value_t* b;
    index_t length_b;
};

}
