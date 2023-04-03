#ifndef CARE_CONCURRENCY_HELPERS_HPP
#define CARE_CONCURRENCY_HELPERS_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <cassert>

#include <moodycamel/readerwriterqueue/readerwriterqueue.h>
#include <moodycamel/concurrentqueue/blockingconcurrentqueue.h>

namespace care{

struct SyncFlag{
    std::atomic<bool> busy{false};
    std::mutex m;
    std::condition_variable cv;

    void setBusy(){
        assert(busy == false);
        busy = true;
    }

    bool isBusy() const{
        return busy;
    }

    void wait(){
        if(isBusy()){
            std::unique_lock<std::mutex> l(m);
            while(isBusy()){
                cv.wait(l);
            }
        }
    }

    void signal(){
        std::unique_lock<std::mutex> l(m);
        busy = false;
        cv.notify_all();
    }        
};

template<class T>
struct WaitableData{
    T data;
    SyncFlag syncFlag;

    void setBusy(){
        syncFlag.setBusy();
    }

    bool isBusy() const{
        return syncFlag.isBusy();
    }

    void wait(){
        syncFlag.wait();
    }

    void signal(){
        syncFlag.signal();
    } 
};


template<class T>
struct SimpleConcurrentQueue{
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cv;

    void push(T item){
        std::lock_guard<std::mutex> lg(mutex);
        queue.emplace(std::move(item));
        cv.notify_one();
    }

    //wait until queue is not empty, then remove first element from queue and return it
    T pop(){
        std::unique_lock<std::mutex> ul(mutex);

        while(queue.empty()){
            cv.wait(ul);
        }

        T item = queue.front();
        queue.pop();
        return item;
    }
};

template<class T>
struct SingleProducerSingleConsumerQueue{
    moodycamel::BlockingReaderWriterQueue<T> queue;

    void push(T item){
        queue.enqueue(item); 
    }

    //wait until queue is not empty, then remove first element from queue and return it
    T pop(){
        T item{};
        queue.wait_dequeue(item);
        return item;
    }
};


template<class T>
struct MultiProducerMultiConsumerQueue{
    moodycamel::BlockingConcurrentQueue<T> queue{};

    void push(T item){
        queue.enqueue(item); 
    }

    //wait until queue is not empty, then remove first element from queue and return it
    T pop(){
        T item;
        queue.wait_dequeue(item);
        return item;
    }
};



}

#endif