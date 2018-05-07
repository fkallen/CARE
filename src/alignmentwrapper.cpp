#include "../inc/alignmentwrapper.hpp"

namespace care{

/*

    ########## SHIFTED HAMMING DISTANCE ##########

*/

void init_SHDhandle(SHDhandle& handle, int deviceId,
                        int max_sequence_length,
                        int max_sequence_bytes,
                        int gpuThreshold){

    handle.timings.reset();

    cuda_init_SHDdata(handle.buffers, deviceId,
                            max_sequence_length,
                            max_sequence_bytes,
                            gpuThreshold);
}

//free buffers
void destroy_SHDhandle(SHDhandle& handle){
    cuda_cleanup_SHDdata(handle.buffers);
}




/*

    ########## SEMI GLOBAL ALIGNMENT ##########

*/

void init_SGAhandle(SGAhandle& handle, int deviceId,
                        int max_sequence_length,
                        int max_sequence_bytes,
                        int gpuThreshold){

    handle.timings.reset();

    cuda_init_SGAdata(handle.buffers, deviceId,
                            max_sequence_length,
                            max_sequence_bytes,
                            gpuThreshold);
}

//free buffers
void destroy_SGAhandle(SGAhandle& handle){
    cuda_cleanup_SGAdata(handle.buffers);
}

}
