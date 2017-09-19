#pragma once

#include <blitzml/sparse_linear/sparse_linear_solver.h>

namespace BlitzML {

class SparseLogRegSolver : public SparseLinearSolver {
  public:
    virtual ~SparseLogRegSolver() { }

  private:
    std::vector<value_t> exp_Aomega;
    std::vector<bool> is_positive_label;
    index_t num_positive_labels;
    bool problem_is_degenerate;
    
    inline value_t compute_prob(index_t j);
    inline value_t compute_x_value(index_t j);

    void initialize_blitz_variables(value_t* initial_conditions);
    void initialize_is_positive_label();
    void initialize_x_variables();
    void initialize_y_and_z_variables();

    void check_for_degenerate_problem();
    void check_for_poor_initialization();

    void update_bias(int max_newton_itr=4);
    void perform_backtracking();
    value_t compute_dual_obj() const;
    void update_subproblem_obj_vals();
    void update_newton_2nd_derivatives(value_t epsilon_to_add=0.);
};

}

