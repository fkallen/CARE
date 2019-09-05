#ifndef CARE_THREADPOOL_HPP
#define CARE_THREADPOOL_HPP

#include <parallel/parallel_task_queue.h>
#include <memory>
#include <mutex>

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
