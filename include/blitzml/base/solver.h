#pragma once 

#include <blitzml/base/common.h>
#include <blitzml/base/working_set.h>
#include <blitzml/base/logger.h>
#include <blitzml/base/timer.h>
#include <blitzml/dataset/dataset.h>
#include <blitzml/base/scheduler.h>

namespace BlitzML {

class Solver {

  public:
    Solver() :
      data(NULL),
      params(NULL),
      tolerance_(0.001),
      max_time_(3.15569e9),
      min_time_(-1.0),
      max_iterations_(100000),
      verbose_(false),
      use_screening_(true),
      use_working_sets_(true),
      log_vectors_(false),
      suppress_warnings_(false)
    { }

    virtual ~Solver() { debug("delete solver"); }

    void solve(const Dataset *data,
               const Parameters *params,
               value_t *result,
               char *solution_status,
               const char *log_directory);

    void set_tolerance(value_t val) { tolerance_ = val; }
    value_t tolerance() const { return tolerance_; }
    void set_max_time(value_t val) { max_time_ = val; }
    value_t max_time() const { return max_time_; }
    void set_min_time(value_t val) { min_time_ = val; }
    value_t min_time() const { return min_time_; }
    void set_max_iterations(unsigned val) { max_iterations_ = val; }
    unsigned max_iterations() const { return max_iterations_; }
    void set_verbose(bool val) { verbose_ = val; }
    bool verbose() const { return verbose_; }
    void set_use_screening(bool val) { use_screening_ = val; }
    bool use_screening() const { return use_screening_; }
    void set_use_working_sets(bool val) { use_working_sets_ = val; }
    bool use_working_sets() const { return use_working_sets_; }
    void set_log_vectors(bool val) { log_vectors_ = val; }
    bool log_vectors() const { return log_vectors_; }
    void set_suppress_warnings(bool val) { suppress_warnings_ = val; }
    bool suppress_warnings() const { return suppress_warnings_; }


  protected:

    virtual void solve_subproblem() = 0;
    void set_initial_subproblem_state(SubproblemState& initial_state) const;
    bool check_sufficient_dual_progress(
        const SubproblemState& initial_state, value_t epsilon) const;
    virtual value_t norm_diff_sq_z_initial_x(const SubproblemState& initial_state) const = 0;

    virtual value_t compute_alpha() = 0;

    virtual void update_y() = 0;

    virtual unsigned short compute_priority_value(index_t i) const = 0;

    virtual value_t compute_dual_obj() const = 0;
    virtual value_t compute_primal_obj_x() const = 0;
    virtual value_t compute_primal_obj_y() const = 0;

    virtual value_t compute_d_sq() const;

    virtual void screen_components() = 0;

    virtual void set_subproblem_parameters();
    virtual void setup_subproblem_no_working_sets();

    virtual void record_subproblem_size();
    virtual size_t size_of_component(index_t i) const = 0;

    virtual index_t initial_problem_size() const = 0;

    virtual void initialize(value_t *result) = 0;
    virtual value_t strong_convexity_constant() const;

    virtual void log_variables(Logger &logger) const { };

    virtual void fill_result(value_t *result) const = 0;


    const Dataset* data;
    const Parameters* params;
    WorkingSet ws;

    std::vector<value_t> x;
    std::vector<value_t> y;

    ObjectiveValues obj_vals;
    Scheduler scheduler;
    SubproblemParameters subproblem_params;
    std::vector<index_t> prioritized_indices_ar;
    std::vector<unsigned char> priority_values;

    std::vector<CapsuleParams> capsule_candidates;
    std::vector<value_t> tau_candidates;
    std::vector<value_t> left_beta_candidates;
    std::vector<value_t> right_beta_candidates;
    std::vector<value_t> problem_size_diffs;

    index_t min_size_working_set;
    index_t max_size_working_set;
    value_t alpha;
    unsigned iteration_number;
    unsigned long num_updates;
    index_t num_components;


  private:
    enum ConvergenceStatus {
      NOT_CONVERGED,
      REACHED_STOPPING_TOLERANCE,
      EXCEEDED_MAX_ITERATIONS,
      EXCEEDED_TIME_LIMIT,
      REACHED_MACHINE_PRECISION
    };

    Timer solver_timer;
    Timer subproblem_solve_timer;
    Logger solver_logger;
    value_t duality_gap_last_iteration;
    value_t dual_objective_last_iteration;

    value_t tolerance_;
    value_t max_time_;
    value_t min_time_;
    value_t last_log_gap;
    unsigned max_iterations_;
    bool verbose_;
    bool use_screening_;
    bool use_working_sets_;
    bool log_vectors_;
    bool suppress_warnings_;


    void initialize_priority_vectors();
    void update_alpha();
    void update_obj_vals();
    void setup_subproblem();
    void update_priority_values();
    bool should_force_log();
    void record_subproblem_progress();
    void log_info(bool force_log=false);
    Solver::ConvergenceStatus get_convergence_status();
    void set_status_message(char* status_message, Solver::ConvergenceStatus status);

}; 

} // namespace BlitzML
