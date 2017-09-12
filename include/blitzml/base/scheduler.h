#pragma once 

#include <blitzml/base/common.h>
#include <blitzml/base/capsule.h>

namespace BlitzML {

class Scheduler {

  public:
    Scheduler() { reset(); }

    virtual ~Scheduler() { }

    void capsule_candidates(std::vector<CapsuleParams>& capsule_candidates) const;
    void set_problem_sizes(const std::vector<value_t> &problem_sizes);

    void set_strong_convexity_constant(value_t gamma) { this->gamma = gamma; }
    void set_current_duality_gap(value_t Delta) { current_Delta = Delta; }
    void set_current_dsq(value_t dsq) { current_dsq = dsq; }

    void optimize_xi_and_epsilon(unsigned short &best_capsule_ind, value_t &best_epsilon, double &best_time_limit);

    void record_subproblem_size(value_t xi, 
                                value_t subproblem_size);
    void record_overhead_time(double time);
    void record_subproblem_progress(value_t new_Delta, 
                                    value_t subproblem_Delta,
                                    double subproblem_time,
                                    double alpha);

    double overhead_time_estimate();

    void reset();


  private:
    std::vector<value_t> epsilon_candidates;
    std::vector<value_t> xi_candidates;
    std::vector<value_t> problem_size_candidates;

    std::vector<value_t> C_progress_estimates;
    std::vector<value_t> C_solve_time_estimates;
    std::vector<value_t> C_overhead_time_estimates;

    value_t chosen_xi;
    value_t chosen_subproblem_size;

    value_t current_Delta;
    value_t current_dsq;
    value_t gamma;

    value_t subproblem_time_complexity(value_t epsilon);
    value_t estimate_C_progress();
    void set_epsilon_candidates(const std::vector<value_t> &epsilons);

};

}
