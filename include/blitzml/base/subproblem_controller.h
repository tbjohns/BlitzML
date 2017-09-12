#pragma once

#include <blitzml/base/common.h>
#include <blitzml/base/timer.h>

namespace BlitzML {

class SubproblemController {
  public:
    SubproblemController(const SubproblemParameters &subproblem_params);

    virtual ~SubproblemController() { }

    void start_iteration();

    bool should_compute_duality_gap();
    bool should_terminate(const ObjectiveValues &obj_vals, bool sufficient_dual_progress);

  private:
    Timer timer;
    const SubproblemParameters subproblem_params;
    int itr;
    int scheduled_termination_check;

    SubproblemController();
};

} // namespace BlitzML
