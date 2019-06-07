#ifndef SIMPLEALLOCATION_HPP
#define SIMPLEALLOCATION_HPP

#ifdef __NVCC__

namespace detail{

	enum class DataLocation {Host, PinnedHost, Device};

	template<DataLocation location, class T>
	struct SimpleAllocator;

	template<class T>
	struct SimpleAllocator<DataLocation::Host, T>{
		T* allocate(size_t elements){
			T* ptr{};
			ptr = new T[elements];
			return ptr;
		}

		void deallocate(T* ptr){
			delete [] ptr;
		}
	};

	template<class T>
	struct SimpleAllocator<DataLocation::PinnedHost, T>{
		T* allocate(size_t elements){
			T* ptr{};
			cudaMallocHost(&ptr, elements * sizeof(T)); CUERR;
			return ptr;
		}

		void deallocate(T* ptr){
			cudaFreeHost(ptr); CUERR;
		}
	};

	template<class T>
	struct SimpleAllocator<DataLocation::Device, T>{
		T* allocate(size_t elements){
			T* ptr;
			cudaMalloc(&ptr, elements * sizeof(T)); CUERR;
			return ptr;
		}

		void deallocate(T* ptr){
			cudaFree(ptr); CUERR;
		}
	};

    template<DataLocation location, class T>
    struct SimpleAllocationAccessor;

    template<class T>
    struct SimpleAllocationAccessor<DataLocation::Host, T>{
        static const T& access(const T* array, size_t i){
            return array[i];
        }
        static T& access(T* array, size_t i){
            return array[i];
        }
    };

    template<class T>
    struct SimpleAllocationAccessor<DataLocation::PinnedHost, T>{
        static const T& access(const T* array, size_t i){
            return array[i];
        }
        static T& access(T* array, size_t i){
            return array[i];
        }
    };

    template<class T>
    struct SimpleAllocationAccessor<DataLocation::Device, T>{
        //cannot access device memory on host, return default value
        static T access(const T* array, size_t i){
            return T{};
        }
        static T access(T* array, size_t i){
            return T{};
        }
    };

	template<DataLocation location, class T>
	struct SimpleAllocation{
		using Allocator = SimpleAllocator<location, T>;
        using Accessor = SimpleAllocationAccessor<location, T>;

		T* data_{};
		size_t size_{};
		size_t capacity_{};

		SimpleAllocation() : SimpleAllocation(0){}
		SimpleAllocation(size_t size){
			resize(size);
		}

        SimpleAllocation(const SimpleAllocation&) = delete;
        SimpleAllocation& operator=(const SimpleAllocation&) = delete;

        SimpleAllocation(SimpleAllocation&& rhs) noexcept{
            *this = std::move(rhs);
        }

        SimpleAllocation& operator=(SimpleAllocation&& rhs) noexcept{
            Allocator alloc;
            alloc.deallocate(data_);

            data_ = rhs.data_;
            size_ = rhs.size_;
            capacity_ = rhs.capacity_;

            rhs.data_ = nullptr;
            rhs.size_ = 0;
            rhs.capacity_ = 0;

            return *this;
        }

        ~SimpleAllocation(){
            Allocator alloc;
            alloc.deallocate(data_);
        }

        friend void swap(SimpleAllocation& l, SimpleAllocation& r) noexcept{
    		using std::swap;

    		swap(l.data_, r.data_);
    		swap(l.size_, r.size_);
    		swap(l.capacity_, r.capacity_);
    	}

        T& operator[](size_t i){
            return Accessor::access(get(), i);
        }

        const T& operator[](size_t i) const{
            return Accessor::access(get(), i);
        }

        T& at(size_t i){
            if(i < size()){
                return Accessor::access(get(), i);
            }else{
                throw std::out_of_range("SimpleAllocation::at out-of-bounds access.");
            }
        }

        const T& at(size_t i) const{
            if(i < size()){
                return Accessor::access(get(), i);
            }else{
                throw std::out_of_range("SimpleAllocation::at out-of-bounds access.");
            }
        }

        T* operator+(size_t i) const{
            return get() + i;
        }

        operator T*(){
            return get();
        }

        operator const T*() const{
            return get();
        }


        //size is number of elements of type T
		void resize(size_t newsize){
			if(capacity_ < newsize){
				Allocator alloc;
				alloc.deallocate(data_);
				data_ = alloc.allocate(newsize);
				capacity_ = newsize;
			}
			size_ = newsize;
		}

		T* get() const{
			return data_;
		}

		size_t size() const{
			return size_;
		}

		size_t& sizeRef(){
			return size_;
		}

        size_t sizeInBytes() const{
            return size_ * sizeof(T);
        }
	};

}

template<class T>
using SimpleAllocationHost = detail::SimpleAllocation<detail::DataLocation::Host, T>;

template<class T>
using SimpleAllocationPinnedHost = detail::SimpleAllocation<detail::DataLocation::PinnedHost, T>;

template<class T>
using SimpleAllocationDevice = detail::SimpleAllocation<detail::DataLocation::Device, T>;

#endif

#endif
