#include "../inc/alignment.hpp"

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








// split substitutions in alignment into deletion + insertion
int split_subs(AlignResult& alignment, const char* subject){
	auto& ops = alignment.operations;
	int splitted_subs = 0;
	for(auto it = ops.begin(); it != ops.end(); it++){
		if(it->type == ALIGNTYPE_SUBSTITUTE){
			AlignOp del = *it;
			del.base = subject[it->position];
			del.type = ALIGNTYPE_DELETE;

			AlignOp ins = *it;
			ins.type = ALIGNTYPE_INSERT;

			it = ops.erase(it);
			it = ops.insert(it, del);
			it = ops.insert(it, ins);
			splitted_subs++;
		}
	}
	return splitted_subs;
};


}
