#pragma once 

#include <blitzml/base/common.h>
#include <blitzml/dataset/dataset.h>

namespace BlitzML {

template <typename data_t>
class DenseColumn : public Column {
  public:
    DenseColumn(const data_t *values, index_t length);

    virtual ~DenseColumn() { }

    value_t inner_product(const std::vector<value_t> &vec) const;
    value_t weighted_inner_product(const std::vector<value_t> &vec, 
                                   const std::vector<value_t> &weights) const;
    value_t weighted_norm_sq(const std::vector<value_t> &weights) const;

    void add_multiple(std::vector<value_t> &target, value_t scalar) const;
    void weighted_add_multiple(
        std::vector<value_t> &target, std::vector<value_t> &weights, value_t scalar) const;

    value_t sum() const;
    value_t mean() const;
    value_t l2_norm_sq() const;

  private:
    const data_t *values;

    DenseColumn();
};

size_t coordinates2dense_array_index(index_t row, index_t col, index_t height);

template <typename data_t>
class DenseDataset : public Dataset {
  public:
    DenseDataset(const data_t *data, index_t height, index_t width, 
                 const value_t *b, index_t length_b);

    virtual ~DenseDataset();

    const Column* column(index_t j) const { return A_cols[j]; }

    bool any_columns_overlapping_in_submatrix(
        index_t first_column_submatrix, index_t end_column_submatrix) const;

  private:
    std::vector<DenseColumn<data_t>* > A_cols;

    DenseDataset();
};


} // namespace BlitzML
