#pragma once

#include <blitzml/sparse_linear/sparse_linear_solver.h>

namespace BlitzML {

class LassoSolver : public SparseLinearSolver {
  public:
    virtual ~LassoSolver() { }

  protected: 
    value_t compute_dual_obj() const;
    value_t compute_primal_obj_x() const;
    value_t compute_primal_obj_y() const;
    void update_bias(int max_newton_itr=4);
    void setup_proximal_newton_problem();
    void perform_backtracking();
    value_t update_coordinates_in_working_set();

    void update_bias_subproblem() { }
    void scale_Delta_Aomega_by_2nd_derivatives() { }
    void unscale_Delta_Aomega_by_2nd_derivatives() { }

    inline value_t update_feature_lasso(index_t working_set_ind);
};


}


