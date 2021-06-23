#ifndef CARE_CUBVECTOR_CUH

#include <cub/util_allocator.cuh>
#include <hpc_helpers.cuh>

namespace care{

    template<class T, int overprovisioningPercent = 0>
    struct CachedDeviceUVector{
    public:
        static_assert(overprovisioningPercent >= 0, "overprovisioningPercent < 0");

        static constexpr size_t getOverprovisionedSize(size_t requiredSize){
            if(overprovisioningPercent <= 0){
                return requiredSize;
            }else{
                const double onePercent = requiredSize / 100.0;
                const size_t extra = onePercent * overprovisioningPercent;
                return requiredSize + std::min(std::size_t(1), extra);
            }            
        }
    private:
        T* data_{};
        size_t size_{};
        size_t capacity_{};
        cub::CachingDeviceAllocator* allocator_{};
    public:

        void setAllocator(cub::CachingDeviceAllocator& alloc){
            allocator_ = &alloc;
        }

        CachedDeviceUVector() {}
        CachedDeviceUVector(cub::CachingDeviceAllocator& alloc) : allocator_(&alloc) {}

        CachedDeviceUVector(size_t size, cudaStream_t stream, cub::CachingDeviceAllocator& alloc)
            : allocator_(&alloc)
        {
            resize(size, stream);
        }

        CachedDeviceUVector(const CachedDeviceUVector& rhs) = delete;
        CachedDeviceUVector& operator=(const CachedDeviceUVector&) = delete;

        CachedDeviceUVector(CachedDeviceUVector&& rhs) noexcept{
            *this = std::move(rhs);
        }

        CachedDeviceUVector& operator=(CachedDeviceUVector&& rhs) noexcept{

            if(data_ != nullptr){
                allocator_->DeviceFree(data_);
            }            

            data_ = rhs.data_;
            size_ = rhs.size_;
            capacity_ = rhs.capacity_;
            allocator_ = rhs.allocator_;

            rhs.data_ = nullptr;
            rhs.size_ = 0;
            rhs.capacity_ = 0;

            return *this;
        }

        ~CachedDeviceUVector(){
            destroy();
        }

        friend void swap(CachedDeviceUVector& l, CachedDeviceUVector& r) noexcept{
            using std::swap;

            swap(l.data_, r.data_);
            swap(l.size_, r.size_);
            swap(l.capacity_, r.capacity_);
            swap(l.allocator_, r.allocator_);
        }
        
        void destroy(){
            if(data_ != nullptr){
                allocator_->DeviceFree(data_);
            }
            size_ = 0;
            capacity_ = 0;
            data_ = nullptr;
        }

        T& operator[](size_t i){
            return data()[i];
        }

        const T& operator[](size_t i) const{
            return data()[i];
        }

        //return true if reallocation occured
        //memory content is unspecified after operation
        bool reserveWithoutCopy(size_t newcapacity, cudaStream_t stream){

            if(capacity_ < newcapacity){
                if(data_ != nullptr){
                    allocator_->DeviceFree(data_);
                }
                allocator_->DeviceAllocate((void**)&data_, sizeof(T) * newcapacity, stream); CUERR;
                capacity_ = newcapacity;

                return true;
            }else{
                return false;
            }            
        }

        //return true if reallocation occured
        //memory content is unspecified after operation
        bool resizeWithoutCopy(size_t newsize, cudaStream_t stream){
            const size_t newCapacity = getOverprovisionedSize(newsize);
            bool result = reserveWithoutCopy(newCapacity, stream);            
            size_ = newsize;
            return result;
        }

        //return true if reallocation occured
        bool reserve(size_t newcapacity, cudaStream_t stream){

            if(capacity_ < newcapacity){
                T* datanew = nullptr;
                allocator_->DeviceAllocate((void**)&datanew, sizeof(T) * newcapacity, stream); CUERR;
                if(data_ != nullptr){
                    cudaMemcpyAsync(
                        datanew,
                        data_,
                        sizeof(T) * size_,
                        D2D,
                        stream
                    ); CUERR;
                    allocator_->DeviceFree(data_);
                }

                data_ = datanew;
                capacity_ = newcapacity;

                return true;
            }else{
                return false;
            }            
        }

        //return true if reallocation occured
        //memory content in range [oldsize, newsize] is unspecified after operation
        bool resize(size_t newsize, cudaStream_t stream){
            const size_t newCapacity = getOverprovisionedSize(newsize);
            bool result = reserve(newCapacity, stream);            
            size_ = newsize;
            return result;
        }
        
        //return true if reallocation occured
        bool append(const T* rangeBegin, const T* rangeEnd, cudaStream_t stream){
            const std::size_t rangesize = std::distance(rangeBegin, rangeEnd);
            if(rangesize > 0){
                const std::size_t oldsize = size();
                bool realloc = resize(size() + rangesize, stream);
                cudaMemcpyAsync(
                    data() + oldsize,
                    rangeBegin,
                    sizeof(T) * rangesize,
                    cudaMemcpyDefault,
                    stream
                ); CUERR;

                return realloc;
            }
            return false;
        }

        size_t size() const{
            return size_;
        }

        size_t sizeInBytes() const{
            return size() * sizeof(T);
        }

        size_t capacity() const{
            return capacity_;
        }

        size_t capacityInBytes() const{
            return capacity() * sizeof(T);
        }

        T* data() const noexcept{
            return data_;
        }

        T* begin() const noexcept{
            return data();
        }

        T* end() const noexcept{
            return data() + size();
        }

        bool empty() const noexcept{
            return size() == 0;
        }
    };





















#if 0
    template<class T>
    class DeviceUVectorAllocator{
        virtual T* allocate(size_t elements, cudaStream_t stream){
            T* ptr;
            cudaError_t err = cudaMalloc(&ptr, elements * sizeof(T));
            if(err != cudaSuccess){
                std::cerr << "DeviceUVectorAllocator: Failed to allocate " << (elements) << " * " << sizeof(T) 
                            << " = " << (elements * sizeof(T)) << " bytes\n";

                throw std::bad_alloc();
            }

            assert(ptr != nullptr);
            
            return ptr;
        }

        virtual void deallocate(T* ptr){
            cudaFree(ptr); CUERR;
        }
    };

    template<class T>
    class DeviceUVectorCubCachedAllocator : public DeviceUVectorAllocator<T>{
    private:
        cub::CachingDeviceAllocator* cubAllocator{};
    public:
        DeviceUVectorCubCachedAllocator(cub::CachingDeviceAllocator& alloc) 
            : cubAllocator(alloc){}

        T* allocate(size_t elements, cudaStream_t stream) override {
            T* ptr;
            cudaError_t err = cubAllocator->DeviceAllocate((void**)&ptr, elements * sizeof(T), stream);
            if(err != cudaSuccess){
                std::cerr << "DeviceUVectorCubCachedAllocator: Failed to allocate " << (elements) << " * " << sizeof(T) 
                            << " = " << (elements * sizeof(T)) << " bytes\n";

                throw std::bad_alloc();
            }

            assert(ptr != nullptr);
            
            return ptr;
        }

        void deallocate(T* ptr) override {
            cubAllocator->DeviceFree(ptr); CUERR;
        }
    };

    template<class T, int overprovisioningPercent = 10>
    struct DeviceUVector{
    public:
        static_assert(overprovisioningPercent >= 0, "overprovisioningPercent < 0");

        static constexpr size_t getOverprovisionedSize(size_t requiredSize){
            if(overprovisioningPercent <= 0){
                return requiredSize;
            }else{
                const double onePercent = requiredSize / 100.0;
                const size_t extra = onePercent * overprovisioningPercent;
                return requiredSize + std::min(std::size_t(1), extra);
            }            
        }
    private:
        T* data_{};
        size_t size_{};
        size_t capacity_{};
        DeviceUVectorAllocator* allocator{};
        cudaStream_t stream_{};
    public:

        void setAllocator(DeviceUVectorAllocator& alloc){
            allocator_ = &alloc;
        }


        DeviceUVector(size_t size, cudaStream_t stream, DeviceUVectorAllocator* alloc)
            : allocator_(alloc), stream_(stream)
        {
            resize(size, stream_);
        }

        DeviceUVector(const DeviceUVector& rhs) = delete;
        DeviceUVector& operator=(const DeviceUVector&) = delete;

        DeviceUVector(DeviceUVector&& rhs) noexcept{
            *this = std::move(rhs);
        }

        DeviceUVector& operator=(DeviceUVector&& rhs) noexcept{

            if(data_ != nullptr){
                allocator->deallocate(data_);
            }            

            data_ = rhs.data_;
            size_ = rhs.size_;
            capacity_ = rhs.capacity_;
            allocator_ = rhs.allocator_;
            stream_ = rhs.stream_;

            rhs.data_ = nullptr;
            rhs.size_ = 0;
            rhs.capacity_ = 0;

            return *this;
        }

        ~DeviceUVector(){
            destroy();
        }

        friend void swap(SimpleAllocation& l, SimpleAllocation& r) noexcept{
            using std::swap;

            swap(l.data_, r.data_);
            swap(l.size_, r.size_);
            swap(l.capacity_, r.capacity_);
            swap(l.allocator_, r.allocator_);
            swap(l.stream_, r.stream_);
        }
        
        void destroy(){
            if(data_ != nullptr){
                allocator->deallocate(data_);
            }
            size_ = 0;
            capacity_ = 0;
        }

        T& operator[](size_t i){
            return data()[i];
        }

        const T& operator[](size_t i) const{
            return data()[i];
        }

        //size is number of elements of type T
        //return true if reallocation occured
        bool resizeWithoutCopy(size_t newsize, cudaStream_t stream){
            size_ = newsize;

            if(capacity_ < newsize){
                allocator->deallocate(data_);
                const size_t newCapacity = getOverprovisionedSize(newsize);
                data_ = alloc.allocate(newCapacity);
                capacity_ = newCapacity;

                return true;
            }else{
                return false;
            }            
        }
        
        //reserve enough memory for at least max(newCapacity,newSize) elements, and set size to newSize
        //return true if reallocation occured
        bool reserveAndResize(size_t newCapacity, size_t newSize){
            size_ = newSize;

            newCapacity = std::max(newCapacity, newSize);

            if(capacity_ < newCapacity){
                Allocator alloc;
                alloc.deallocate(data_);
                data_ = alloc.allocate(newCapacity);
                capacity_ = newCapacity;

                return true;
            }else{
                return false;
            }
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
            return size() * sizeof(T);
        }

        size_t capacity() const{
            return capacity_;
        }

        size_t capacityInBytes() const{
            return capacity() * sizeof(T);
        }

        T* data() const noexcept{
            return data_;
        }

        T* begin() const noexcept{
            return data();
        }

        T* end() const noexcept{
            return data() + size();
        }

        bool empty() const noexcept{
            return size() == 0;
        }
    };

#endif

}



#endif