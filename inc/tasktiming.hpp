#ifndef CARE_TASK_TIMING_HPP
#define CARE_TASK_TIMING_HPP

#include <chrono>

namespace care{

    struct TaskTimings{
        std::chrono::duration<double> preprocessingtime{0};
        std::chrono::duration<double> h2dtime{0};
        std::chrono::duration<double> executiontime{0};
        std::chrono::duration<double> d2htime{0};
        std::chrono::duration<double> postprocessingtime{0};
    };

}

#endif
