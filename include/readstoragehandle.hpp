#ifndef CARE_READSTORAGEHANDLE_HPP
#define CARE_READSTORAGEHANDLE_HPP


namespace care{

    class CpuReadStorage; //forward declaration
    class GpuReadStorage; //forward declaration

    class ReadStorageHandle{
        
    friend class CpuReadStorage;
    friend class GpuReadStorage;

    public:

        int getId() const noexcept{
            return id;
        }

    private:
        Handle() = default;
        Handle(int i) : id(i){}

        int id;
    };


}


#endif