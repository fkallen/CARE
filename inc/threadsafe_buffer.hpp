#ifndef THREADSAFE_BUFFER_HPP
#define THREADSAFE_BUFFER_HPP

#include <condition_variable>
#include <mutex>
#include <array>

/*
A thread-safe circular buffer with fixed capacity
*/

template<class T, size_t capacity_>
struct ThreadsafeBuffer {

	const T defaultValue;
	const size_t capacity = capacity_;

	std::int64_t curIndex = 0;
	size_t nElem = 0;
	std::array<T, capacity_> buffer;

	std::mutex mutex;
	std::condition_variable condvar;

	bool noMoreInserts = false;

	std::uint64_t addWait = 0;
	std::uint64_t addNoWait = 0;
	std::uint64_t getWait = 0;
	std::uint64_t getNoWait = 0;

	ThreadsafeBuffer(){}

	void add(T data)
	{
		std::unique_lock<std::mutex> lock(mutex);
		if (nElem >= capacity_) {
			addWait++;
			condvar.wait(lock);
		}else{
			addNoWait++;
		}
		nElem++;
		buffer[curIndex] = data;
		curIndex = (curIndex + 1) % capacity_;
		condvar.notify_all();
	}

	T get()
	{
		std::unique_lock<std::mutex> lock(mutex);
		if (nElem == 0 && !noMoreInserts) {
			getWait++;
			condvar.wait(lock);
		}else{
			getNoWait++;
		}
		if (nElem == 0 && noMoreInserts) {
			return defaultValue;
		}else{
			int index = (curIndex - nElem + capacity_) % capacity_;
			T retVal = buffer[index];
			nElem--;
			condvar.notify_all();
			return retVal;
		}
	}

	void done()
	{
		std::unique_lock<std::mutex> lock(mutex);
		noMoreInserts = true;
		//important! threads may wait until noMoreInserts == true
		condvar.notify_all();
	}

	// call only if no other thread is using this buffer
	void reset()
	{
		noMoreInserts = false;
		curIndex = 0;
		nElem = 0;
		addWait = 0;
		addNoWait = 0;
		getWait = 0;
		getNoWait = 0;
	}

	ThreadsafeBuffer(const ThreadsafeBuffer &other){
		*this = other;
	}

	ThreadsafeBuffer& operator=(const ThreadsafeBuffer& other)
	{
		curIndex = other.curIndex;
		nElem = other.nElem;
		buffer = other.buffer;
		noMoreInserts = other.noMoreInserts;
		addWait = other.addWait;
		addNoWait = other.addNoWait;
		getWait = other.getWait;
		getNoWait = other.getNoWait;

		return *this;
	}


};

#endif
