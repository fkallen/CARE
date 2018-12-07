#ifndef THREADSAFE_BUFFER_HPP
#define THREADSAFE_BUFFER_HPP

#include <condition_variable>
#include <mutex>
#include <array>

namespace care{

/*
A thread-safe circular buffer with fixed capacity
*/

template<class T, size_t capacity_>
struct ThreadsafeBuffer {
	using Value_t = T;

	struct PopResult{
		Value_t value;
		bool foreverEmpty;
	};

	Value_t defaultValue;
	const size_t capacity = capacity_;

	std::int64_t curIndex = 0;
	size_t nElem = 0;
	std::vector<Value_t> buffer;

	std::mutex mutex;
	std::condition_variable condvar;

	bool noMoreInserts = false;

	std::uint64_t addWait = 0;
	std::uint64_t addNoWait = 0;
	std::uint64_t getWait = 0;
	std::uint64_t getNoWait = 0;

	ThreadsafeBuffer(){
        defaultValue = Value_t{};
		buffer.resize(capacity_);
	}

	void printWaitStatistics(){
		std::cout << "addWait: " << addWait << ' '
				  << "addNoWait: " << addNoWait << ' '
				  << "getWait: " << getWait << ' '
				  << "getNoWait: " << getNoWait << std::endl;
	}

	void add(const Value_t& data)
	{
		std::unique_lock<std::mutex> lock(mutex);
		if (nElem >= capacity_) {
			addWait++;
			while(nElem >= capacity_)
				condvar.wait(lock);
		}else{
			addNoWait++;
		}
		nElem++;
		buffer[curIndex] = data;
		curIndex = (curIndex + 1) % capacity_;
		condvar.notify_one();
	}

	void add(Value_t&& data)
	{
		std::unique_lock<std::mutex> lock(mutex);
		if (nElem >= capacity_) {
			addWait++;
			while(nElem >= capacity_)
				condvar.wait(lock);
		}else{
			addNoWait++;
		}
		nElem++;
		buffer[curIndex] = std::move(data);
		curIndex = (curIndex + 1) % capacity_;
		condvar.notify_one();
	}

	Value_t get()
	{
		std::unique_lock<std::mutex> lock(mutex);
		if (nElem == 0 && !noMoreInserts) {
			getWait++;
			while(nElem == 0 && !noMoreInserts)
				condvar.wait(lock);
		}else{
			getNoWait++;
		}
		if (nElem == 0 && noMoreInserts) {
			return defaultValue;
		}else{
			int index = (curIndex - nElem + capacity_) % capacity_;
			const Value_t& retVal = buffer[index];
			if(nElem == 0) std::cout << "error" << std::endl;
			nElem--;

			condvar.notify_one();
			return retVal;
		}
	}

	PopResult getNew()
	{
		std::unique_lock<std::mutex> lock(mutex);
		if (nElem == 0 && !noMoreInserts) {
			getWait++;
			while(nElem == 0 && !noMoreInserts)
				condvar.wait(lock);
		}else{
			getNoWait++;
		}
		if (nElem == 0 && noMoreInserts) {
			PopResult result;
			result.foreverEmpty = true;
			return result;
		}else{
			PopResult result;
			result.foreverEmpty = false;

			int index = (curIndex - nElem + capacity_) % capacity_;
			result.value = std::move(buffer[index]);

			if(nElem == 0) std::cout << "error" << std::endl;

			nElem--;

			condvar.notify_one();
			return result;
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

	ThreadsafeBuffer(const ThreadsafeBuffer&) = default;
	ThreadsafeBuffer(ThreadsafeBuffer&&) = default;
	ThreadsafeBuffer& operator=(const ThreadsafeBuffer&) = default;
	ThreadsafeBuffer& operator=(ThreadsafeBuffer&&) = default;


};


}

#endif
