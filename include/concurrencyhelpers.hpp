#ifndef CARE_CONCURRENCY_HELPERS_HPP
#define CARE_CONCURRENCY_HELPERS_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <cassert>

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




}

#endif