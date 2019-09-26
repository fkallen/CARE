#ifndef CARE_THREADPOOL_HPP
#define CARE_THREADPOOL_HPP

#include <parallel/parallel_task_queue.h>
#include <memory>
#include <mutex>
#include <condition_variable>

namespace care{

struct ThreadPool{
    using task_type = am::parallel_queue::task_type;

    ThreadPool()
        : pq(std::make_unique<am::parallel_queue>()){
    }

    void enqueue(const task_type& t){
        pq->enqueue(t);
    }

    void enqueue(task_type&& t){
        pq->enqueue(std::move(t));
    }

    /*
        loopBody(begin, end, threadId) must be equivalent to

        for(Index_t i = begin; i < end; i++){
            doStuff(threadId)
        }
    */
    template<class Index_t, class Func>
    void parallelFor(Index_t begin, Index_t end, Func&& loop){
        parallelFor(begin, end, std::forward<Func>(loop), getConcurrency());
    }

    template<class Index_t, class Func>
    void parallelFor(Index_t begin, Index_t end, Func&& loop, std::size_t numThreads){
        constexpr bool waitForCompletion = true;

        parallelFor_impl<waitForCompletion>(begin, end, std::forward<Func>(loop), numThreads);
    }

    template<class Index_t, class Func>
    void parallelForNoWait(Index_t begin, Index_t end, Func&& loop){
        parallelForNoWait(begin, end, std::forward<Func>(loop), getConcurrency());
    }

    template<class Index_t, class Func>
    void parallelForNoWait(Index_t begin, Index_t end, Func&& loop, std::size_t numThreads){
        constexpr bool waitForCompletion = false;

        parallelFor_impl<waitForCompletion>(begin, end, std::forward<Func>(loop), numThreads);
    }

    void wait(){
        pq->wait();
    }

    //don't call this in a situation where another thread could insert work
    void setConcurrency(int numThreads){
        pq->wait();

        pq.reset(new am::parallel_queue(numThreads));
    }

    int getConcurrency() const{
        return pq->concurrency();
    }

private:

    template<bool waitForCompletion, class Index_t, class Func>
    void parallelFor_impl(Index_t firstIndex, Index_t lastIndex, Func&& loop, std::size_t numThreads){
        std::mutex m;
        std::condition_variable cv;
        std::size_t finishedWork = 0;
        std::size_t startedWork = 0;

        auto work = [&, func = std::forward<Func>(loop)](Index_t begin, Index_t end, int threadId){
            func(begin, end, threadId);

            if(waitForCompletion){
                std::lock_guard<std::mutex> lg(m);
                finishedWork++;
                cv.notify_one();
            }            
        };

        Index_t totalIterations = lastIndex - firstIndex;
        if(totalIterations > 0){
            const Index_t chunks = numThreads;
            const Index_t chunksize = totalIterations / chunks;
            const Index_t leftover = totalIterations % chunks;

            Index_t begin = firstIndex;
            Index_t end = begin + chunksize;
            for(Index_t c = 0; c < chunks-1; c++){
                if(c < leftover){
                    end++;
                }

                if(end-begin > 0){
                    startedWork++;

                    enqueue([begin, end, c, work](){
                        work(begin, end, c);
                    });

                    begin = end;
                    end += chunksize;
                }                
            }

            if(end-begin > 0){
                startedWork++;
                work(begin, end, chunks-1);                
            }

            if(waitForCompletion){
                //std::cerr << "Wait for completion " << finishedWork << " / " << startedWork << "\n";
                std::unique_lock<std::mutex> ul(m);
                if(finishedWork != startedWork){
                    //std::cerr << "Waiting\n";
                    cv.wait(ul, [&](){return finishedWork == startedWork;});
                    //std::cerr << "No longer waiting\n";
                }
            }
        }   
    }

    std::unique_ptr<am::parallel_queue> pq;
};



extern ThreadPool threadpool;

}

#endif
