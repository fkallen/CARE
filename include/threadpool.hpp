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
        : pq(std::make_unique<am::parallel_queue>()),
            m{}{

    }

    void enqueue(const task_type& t){
        std::unique_lock<std::mutex> l(m);
        pq->enqueue(t);
    }

    void enqueue(task_type&& t){
        std::unique_lock<std::mutex> l(m);
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
        std::mutex m;
        std::condition_variable cv;
        std::size_t finishedWork = 0;
        std::size_t startedWork = 0;

        auto work = [&, func = std::forward<Func>(loop)](Index_t begin, Index_t end, int threadId){
            func(begin, end, threadId);

            {
                std::lock_guard<std::mutex> lg(m);
                finishedWork++;
                cv.notify_one();
            }            
        };

        Index_t totalIterations = end - begin;
        if(totalIterations > 0){
            const std::size_t chunks = getConcurrency();
            const Index_t chunksize = totalIterations / chunks;
            const Index_t leftover = totalIterations % chunks;

            Index_t begin = 0;
            Index_t end = chunksize;
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

            std::unique_lock<std::mutex> ul(m);
            if(finishedWork != startedWork){
                cv.wait(ul, [&](){return finishedWork == startedWork;});
            }
        }   
    }

    void wait(){
        pq->wait();
    }

    void setConcurrency(int numThreads){
        std::unique_lock<std::mutex> l(m);
        pq->wait();

        pq.reset(new am::parallel_queue(numThreads));
    }

    int getConcurrency() const{
        return pq->concurrency();
    }

    std::unique_ptr<am::parallel_queue> pq;
    std::mutex m;
};



extern ThreadPool threadpool;

}

#endif
