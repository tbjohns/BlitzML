#pragma once

#include <blitzml/base/common.h>
#include <blitzml/base/solver.h>
#include <blitzml/base/logger.h>
#include <blitzml/smooth_loss/smooth_loss.h>

#include <vector>

namespace BlitzML {


class SparseLinearSolver : public Solver {
  public:
    SparseLinearSolver() : loss_function(NULL) { }

    virtual ~SparseLinearSolver() { 
      delete_loss_function();
    }

    value_t compute_max_l1_penalty(const Dataset *data, const Parameters *params);

  protected:

    std::vector<value_t> omega;
    std::vector<value_t> inv_lipschitz_cache;
    std::vector<value_t> col_means_cache;
    std::vector<const Column*> A_cols;
    std::vector<value_t> ATx;
    std::vector<value_t> Delta_omega;
    std::vector<value_t> Delta_Aomega;
    std::vector<value_t> inv_newton_2nd_derivative_cache;
    std::vector<value_t> col_ip_newton_2nd_derivative_cache;
    std::vector<value_t> ATy;
    std::vector<value_t> ATz;
    std::vector<value_t> z;
    std::vector<value_t> Aomega;  
    std::vector<value_t> newton_2nd_derivatives;
    std::vector<index_t> screen_indices_map;

    SmoothLoss* loss_function;

    value_t l1_penalty;
    value_t sum_x;
    value_t sum_newton_2nd_derivatives;
    value_t Delta_bias;
    value_t kappa_x;
    value_t kappa_z;
    value_t bias;
    value_t max_inv_lipschitz_cache;

    value_t change2;

    index_t num_examples;

    static const int MAX_BACKTRACK_STEPS = 25;

    bool z_match_y;
    bool z_match_x;

    bool use_bias;

    // objective values:
    virtual value_t compute_dual_obj() const;
    virtual value_t compute_primal_obj_x() const;
    virtual value_t compute_primal_obj_y() const;
    virtual void update_subproblem_obj_vals();

    // subproblem:
    void solve_subproblem();
    virtual value_t update_coordinates_in_working_set();
    value_t update_feature(index_t working_set_ind);
    virtual void perform_backtracking();
    void set_initial_subproblem_z();
    void update_z();
    value_t norm_diff_sq_z_initial_x(const SubproblemState& initial_state) const;
    virtual void scale_Delta_Aomega_by_2nd_derivatives();
    virtual void unscale_Delta_Aomega_by_2nd_derivatives();
    virtual void setup_proximal_newton_problem();
    void set_max_and_min_num_cd_itr(int &max_num_cd_itr, int &min_num_cd_itr) const;
    void update_kappa_x();
    value_t compute_cd_threshold() const;
    virtual void update_bias(int max_newton_itr=5);
    virtual void update_bias_subproblem();
    value_t perform_newton_update_on_bias();
    value_t compute_backtracking_step_size_derivative(value_t step_size) const;
    void update_x(value_t step_size=0.);
    virtual void update_newton_2nd_derivatives(value_t epsilon_to_add=0.);
    void setup_subproblem_no_working_sets();

    // y updates:
    value_t compute_alpha();
    void update_non_working_set_gradients();
    value_t compute_alpha_for_feature(index_t j) const;
    void update_y();

    // working set:
    unsigned short compute_priority_value(index_t i) const;

    size_t size_of_component(index_t i) const;

    // screening:
    void screen_components();
    bool mark_components_to_screen(std::vector<bool>& should_screen) const;
    void apply_screening_result(const std::vector<bool>& should_screen);

    // initialization:
    index_t initial_problem_size() const;
    void initialize(value_t *result);
    value_t strong_convexity_constant() const;
    void deserialize_parameters();
    void set_data_dimensions();
    virtual void initialize_blitz_variables(value_t* initial_conditions);
    void initialize_model(value_t* initial_conditions);
    void compute_Aomega();
    void cache_feature_info();
    void initialize_proximal_newton_variables();

    // misc:
    void log_variables(Logger &logger) const;
    void fill_result(value_t *result) const;
    void delete_loss_function();
};

} // namespace BlitzML

