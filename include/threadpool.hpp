#ifndef CARE_THREADPOOL_HPP
#define CARE_THREADPOOL_HPP

#include <parallel/parallel_task_queue.h>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <iostream>

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
        //parallelForNoWait(begin, end, std::forward<Func>(loop), getConcurrency());
    }

    template<class Index_t, class Func>
    void parallelForNoWait(Index_t begin, Index_t end, Func&& loop, std::size_t numThreads){
        constexpr bool waitForCompletion = false;
        assert(false);

        //parallelFor_impl<waitForCompletion>(begin, end, std::forward<Func>(loop), numThreads);
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

    /*
        for(auto i = firstIndex; i < lastIndex; i++)
            loop(i)
    */
    template<bool waitForCompletion, class Index_t, class Func>
    void parallelFor_impl(Index_t firstIndex, Index_t lastIndex, Func&& loop, std::size_t numThreads){
        //2 debug variables
        volatile int initialNumRunningParallelForWithWaiting = numRunningParallelForWithWaiting;
        volatile int initialNumUnfinishedParallelForChunks = numUnfinishedParallelForChunks;

        if(waitForCompletion){
            ++numRunningParallelForWithWaiting;
        }

        std::mutex m;
        std::condition_variable cv;
        std::size_t finishedWork = 0;
        std::size_t startedWork = 0;
        std::size_t enqueuedWork = 0;

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
                    if(waitForCompletion){
                        enqueuedWork++;
                        ++numUnfinishedParallelForChunks;

                        enqueue([&, begin, end, c](){
                            {
                                std::lock_guard<std::mutex> lg(m);
                                startedWork++;
                            }

                            loop(begin, end, c);

                            --numUnfinishedParallelForChunks;

                            if(waitForCompletion){
                                std::lock_guard<std::mutex> lg(m);
                                finishedWork++;
                                cv.notify_one();
                            }
                        });
                    }else{
                        ++numUnfinishedParallelForChunks;

                        enqueue([&, begin, end, c](){
                            loop(begin, end, c);

                            --numUnfinishedParallelForChunks;
                        });
                    }

                    begin = end;
                    end += chunksize;
                }                
            }

            if(end-begin > 0){
                loop(begin, end, chunks-1);                
            }

            if(waitForCompletion){
                //std::cerr << "Wait for completion " << startedWork << " / " << finishedWork << " / " << enqueuedWork << "\n";
                std::unique_lock<std::mutex> ul(m);
                if(finishedWork != enqueuedWork){
                    //std::cerr << "Waiting\n";
                    int waitIter = 0;
                    cv.wait(ul, [&](){
                        constexpr int warningThreshold = 50;
                        constexpr int userinputThreshold = 1000;
                        waitIter++;

                        if(waitIter > warningThreshold){
                            std::cerr << "Iter " << waitIter << ", wait for completion " << startedWork << " / " 
                                                << finishedWork << " / " << enqueuedWork << "\n";
                        }

                        if(waitIter > userinputThreshold){
                            int x;
                            std::cerr << "Iter " << waitIter << ", enter number to continue\n";
                            std::cin >> x;
                        }
                        
                        return finishedWork == enqueuedWork;
                    });
                    //std::cerr << "No longer waiting\n";
                }
            }
        }

        if(waitForCompletion){
            --numRunningParallelForWithWaiting;
        }
    }

    std::unique_ptr<am::parallel_queue> pq;
    std::atomic_int numRunningParallelForWithWaiting{0};
    std::atomic_int numUnfinishedParallelForChunks{0};
};



extern ThreadPool threadpool;

}

#endif
