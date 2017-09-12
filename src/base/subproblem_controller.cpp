#include <blitzml/base/subproblem_controller.h>

namespace BlitzML {

  SubproblemController::SubproblemController(const SubproblemParameters &subproblem_params)
    : subproblem_params(subproblem_params) {
    itr = 0;
    scheduled_termination_check = 10;
    timer.reset();
  }


  void SubproblemController::start_iteration() {
    ++itr;
  }


  bool SubproblemController::should_compute_duality_gap() {
    double elapsed_time = timer.elapsed_time();
    bool compute_gap = false;
    if (elapsed_time > subproblem_params.time_limit) {
      compute_gap = true;
    }
    if (itr >= subproblem_params.max_iterations) {
      compute_gap = true;
    } 
    if (elapsed_time < subproblem_params.min_time) {
      compute_gap = false;
    }

    if (itr >= scheduled_termination_check 
        && subproblem_params.epsilon > 0.) {
      compute_gap = true;
    } 

    if (compute_gap) {
      scheduled_termination_check += 20;
    }

    return compute_gap;
  }


  bool SubproblemController::should_terminate(const ObjectiveValues &obj_vals, bool sufficient_dual_progress) {
    double elapsed_time = timer.elapsed_time();
    if (elapsed_time > subproblem_params.time_limit) {
      return true;
    }
    if (itr >= subproblem_params.max_iterations) {
      return true;
    } 
    if (elapsed_time < subproblem_params.min_time) {
      return false;
    }

    value_t subproblem_duality_gap = obj_vals.primal_obj_z() - obj_vals.dual_obj();
    if (subproblem_duality_gap < subproblem_params.epsilon * obj_vals.duality_gap()) {
      if (sufficient_dual_progress) {
        return true;
      }
    }

    return false;
  }

} // namespace BlitzML
