#ifndef CARE_READSTORAGEHANDLE_HPP
#define CARE_READSTORAGEHANDLE_HPP


namespace care{

    class CpuReadStorage; //forward declaration
    namespace gpu{
        class GpuReadStorage; //forward declaration
    }

    class ReadStorageHandle{

        friend class CpuReadStorage;
        friend class gpu::GpuReadStorage;

    public:

        int getId() const noexcept{
            return id;
        }

    private:
        ReadStorageHandle() = default;
        ReadStorageHandle(int i) : id(i){}

        int id;
    };


}


#endif