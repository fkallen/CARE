#ifndef CARE_THREADPOOL_HPP
#define CARE_THREADPOOL_HPP

#include <parallel/parallel_task_queue.h>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>


#ifdef __NVCC__
#include <gpu/nvtxtimelinemarkers.hpp>
#endif

namespace care{




struct BackgroundThread{
    enum class StopType{FinishAndStop, Stop};

    std::vector<std::function<void()>> tasks{};
    std::mutex m{};
    std::condition_variable consumer_cv{};
    std::condition_variable producer_cv{};
    std::thread thread{};
    std::atomic<bool> stop{false};
    std::atomic<bool> finishRemainingTasks{true};

    BackgroundThread() : stop(false){
    }

    BackgroundThread(bool doStart) : stop(false){
        if(doStart){
            start();
        }
    }

    void start(){
        stop = false;

        thread = std::move(std::thread{
            [&](){
                threadfunc();
            }
        });
    }

    template<class Func>
    void enqueue(Func&& func){
        auto wrapper = [f = std::move(func)]() -> void {
            f();
        };

        {
            std::unique_lock<std::mutex> mylock(m);
            producer_cv.wait(mylock, [&](){return tasks.size() < 16;});
            tasks.emplace_back(std::move(wrapper));
            consumer_cv.notify_one();
        }        
    }

    void threadfunc(){
        while(!stop){
            std::unique_lock<std::mutex> mylock(m);
            //nvtx::push_range("bg thread wait", 7);
            consumer_cv.wait(mylock, [&](){return !tasks.empty() || stop;});
            //nvtx::pop_range();

            if(!tasks.empty()){
                auto func = std::move(tasks.front());
                tasks.erase(tasks.begin());
                mylock.unlock();
                producer_cv.notify_one();

                func();
            }
        }

        if(stop && finishRemainingTasks){
            for(const auto& func : tasks){
                func();
            }
            tasks.clear();
        }
    }

    void stopThread(StopType type){
        {
            std::unique_lock<std::mutex> mylock(m);

            if(type == StopType::FinishAndStop){
                finishRemainingTasks = true;
            }else{
                finishRemainingTasks = false;
            }
            stop = true;

            consumer_cv.notify_one();
        }

        thread.join();
    }
};








struct ThreadPool{
    using task_type = am::parallel_queue::task_type;

    struct ParallelForHandle{
        struct ParallelForData{
            bool isDone = true;
            std::size_t finishedWork = 0;
            std::size_t enqueuedWork = 0;
            std::mutex mProgress;
            std::condition_variable cvProgress;

            std::mutex mDone;
            std::condition_variable cvDone;

            void wait(){
                if(!isDone){
                    std::unique_lock<std::mutex> l(mDone);
                    while(!isDone){
                        cvDone.wait(l);
                    }
                }
            }

            void signal(){
                std::unique_lock<std::mutex> l(mDone);
                isDone = true;
                cvDone.notify_all();
            }
        };

        std::unique_ptr<ParallelForData> dataPtr;

        ParallelForHandle() : dataPtr(std::make_unique<ParallelForData>()){}
        ParallelForHandle(ParallelForHandle&&) = default;
        ParallelForHandle& operator=(ParallelForHandle&&) = default;

        void wait(){
            dataPtr->wait();
        }

        void signal(){
            dataPtr->signal();
        }
    };

    ThreadPool()
        : pq(std::make_unique<am::parallel_queue>()){
    }

    ThreadPool(int numThreads)
        : pq(std::make_unique<am::parallel_queue>(numThreads)){
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
    void parallelFor(ParallelForHandle& handle, Index_t begin, Index_t end, Func&& loop){
        parallelFor(handle, begin, end, std::forward<Func>(loop), getConcurrency());
    }

    template<class Index_t, class Func>
    void parallelFor(ParallelForHandle& handle, Index_t begin, Index_t end, Func&& loop, std::size_t numThreads){
        constexpr bool waitForCompletion = true;

        parallelFor_impl<waitForCompletion>(handle, begin, end, std::forward<Func>(loop), numThreads, true);
    }

    template<class Index_t, class Func>
    void parallelForNoWait(ParallelForHandle& handle, Index_t begin, Index_t end, Func&& loop){
        parallelForNoWait(handle, begin, end, std::forward<Func>(loop), getConcurrency());
    }

    template<class Index_t, class Func>
    void parallelForNoWait(ParallelForHandle& handle, Index_t begin, Index_t end, Func&& loop, std::size_t numThreads){
        constexpr bool waitForCompletion = false;

        parallelFor_impl<waitForCompletion>(handle, begin, end, std::forward<Func>(loop), numThreads, false);
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
    void parallelFor_impl(ParallelForHandle& handle, 
                        Index_t firstIndex, 
                        Index_t lastIndex, 
                        Func&& loop, 
                        std::size_t numThreads, 
                        bool selfCanParticipate){

        //2 debug variables
        // volatile int initialNumRunningParallelForWithWaiting = numRunningParallelForWithWaiting;
        // volatile int initialNumUnfinishedParallelForChunks = numUnfinishedParallelForChunks;

        // if(waitForCompletion){
        //     ++numRunningParallelForWithWaiting;
        // }

        //assert(waitForCompletion);

        auto pforData = handle.dataPtr.get();

        handle.wait(); //make sure previous parallelfor is finished

        Index_t totalIterations = lastIndex - firstIndex;
        if(totalIterations > 0){
            pforData->isDone = false;
            pforData->finishedWork = 0;
            //std::size_t startedWork = 0;
            pforData->enqueuedWork = 0;

            const Index_t chunks = numThreads;
            const Index_t chunksize = totalIterations / chunks;
            const Index_t leftover = totalIterations % chunks;

            Index_t begin = firstIndex;
            Index_t end = begin + chunksize;

            const Index_t threadpoolChunks = selfCanParticipate ? chunks - 1 : chunks;

            std::vector<std::function<void()>> chunkFunctions;
            chunkFunctions.reserve(threadpoolChunks);

            for(Index_t c = 0; c < threadpoolChunks; c++){
                if(c < leftover){
                    end++;
                }

                if(end-begin > 0){
                    pforData->enqueuedWork++;

                    chunkFunctions.emplace_back([&, begin, end, c, pforData](){

                        loop(begin, end, c);

                        std::lock_guard<std::mutex> lg(pforData->mProgress);
                        pforData->finishedWork++;
                        pforData->cvProgress.notify_one();
                    });

                    begin = end;
                    end += chunksize;
                }                
            }

            pq->enqueue(chunkFunctions.begin(), chunkFunctions.end());

            if(selfCanParticipate && end-begin > 0){
                loop(begin, end, chunks-1);                
            }

            auto waitUntilThreadPoolChunksAreDoneThenSignal = [pforData](){
                if(pforData->finishedWork != pforData->enqueuedWork){
                    std::unique_lock<std::mutex> ul(pforData->mProgress);
                    while(pforData->finishedWork != pforData->enqueuedWork){
                        pforData->cvProgress.wait(ul);
                    }
                }
                pforData->signal();
            };

            if(waitForCompletion){
                waitUntilThreadPoolChunksAreDoneThenSignal();
            }else{
                pq->enqueue(std::move(waitUntilThreadPoolChunksAreDoneThenSignal));
            }
        }

        // if(waitForCompletion){
        //     --numRunningParallelForWithWaiting;
        // }
    }

    std::unique_ptr<am::parallel_queue> pq;
    // std::atomic_int numRunningParallelForWithWaiting{0};
    // std::atomic_int numUnfinishedParallelForChunks{0};
};

// mainly exists because of cuda device lambda limitations
struct ParallelForLoopExecutor{
    ParallelForLoopExecutor(ThreadPool* tp, ThreadPool::ParallelForHandle* handle)
        : threadPool(tp), pforHandle(handle){}

    template<class Index_t, class Func>
    void operator()(Index_t begin, Index_t end, Func&& loopbody){
        threadPool->parallelFor(
            *pforHandle, 
            begin, 
            end, 
            std::move(loopbody)
        );
    }

    int getNumThreads() const{
        return threadPool->getConcurrency();
    }

    ThreadPool* threadPool;
    ThreadPool::ParallelForHandle* pforHandle;
};



//extern ThreadPool threadpool;

}

#endif
