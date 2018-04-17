#ifndef CARE_TASK_TIMING_HPP
#define CARE_TASK_TIMING_HPP

#include <chrono>
#include <iostream>

namespace care{

    struct TaskTimings{
        std::chrono::time_point<std::chrono::system_clock> a,b,c,d;
        std::chrono::duration<double> preprocessingtime{0};
        std::chrono::duration<double> h2dtime{0};
        std::chrono::duration<double> executiontime{0};
        std::chrono::duration<double> d2htime{0};
        std::chrono::duration<double> postprocessingtime{0};

        TaskTimings& operator+=(const TaskTimings& rhs){
            preprocessingtime += rhs.preprocessingtime;
            h2dtime += rhs.h2dtime;
            executiontime += rhs.executiontime;
            d2htime += rhs.d2htime;
            postprocessingtime += rhs.postprocessingtime;
            return *this;
        }

        void reset(){
            preprocessingtime = std::chrono::duration<double>::zero();
            h2dtime = std::chrono::duration<double>::zero();
            executiontime = std::chrono::duration<double>::zero();
            d2htime = std::chrono::duration<double>::zero();
            postprocessingtime = std::chrono::duration<double>::zero();
        }

        void preprocessingBegin(){
            a = std::chrono::system_clock::now();
        }
        void preprocessingEnd(){
            b = std::chrono::system_clock::now();
            preprocessingtime += b-a;
        }
        void h2dBegin(){
            c = std::chrono::system_clock::now();
        }
        void h2dEnd(){
            d = std::chrono::system_clock::now();
            h2dtime += d-c;
        }
        void executionBegin(){
            a = std::chrono::system_clock::now();
        }
        void executionEnd(){
            b = std::chrono::system_clock::now();
            executiontime += b-a;
        }
        void d2hBegin(){
            c = std::chrono::system_clock::now();
        }
        void d2hEnd(){
            d = std::chrono::system_clock::now();
            d2htime += d-c;
        }
        void postprocessingBegin(){
            a = std::chrono::system_clock::now();
        }
        void postprocessingEnd(){
            b = std::chrono::system_clock::now();
            postprocessingtime += b-a;
        }

        friend std::ostream& operator<<(std::ostream& stream, const TaskTimings& tt){
            stream << "Preprocessing: " << tt.preprocessingtime.count() << '\n';
            //stream << "H2D: " << tt.h2dtime.count() << '\n';
            stream << "Execution: " << tt.executiontime.count() << '\n';
            //stream << "D2H: " << tt.d2htime.count() << '\n';
            stream << "Postprocessing: " << tt.postprocessingtime.count() << '\n';
            return stream;
        }
    };

}

#endif
