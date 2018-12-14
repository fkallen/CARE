#ifdef __NVCC__

#include <nvToolsExt.h>

#endif

#ifdef __NVCC__

__inline__
void push_range(const std::string& name, int cid){
	const uint32_t colors_[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff, 0xdeadbeef, 0x12345678, 0xabcdef42 };
	const int num_colors_ = sizeof(colors_)/sizeof(uint32_t);

	int color_id = cid;
	color_id = color_id%num_colors_;
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.colorType = NVTX_COLOR_ARGB;
	eventAttrib.color = colors_[color_id];
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
	eventAttrib.message.ascii = name.c_str();
	nvtxRangePushEx(&eventAttrib);
	//std::cout << "push " << name << std::endl;
}

__inline__
void pop_range(const std::string& name){
	nvtxRangePop();
	//std::cout << "pop " << name << std::endl;
}

void pop_range(){
	nvtxRangePop();
	//std::cout << "pop " << std::endl;
}

#else

__inline__
void push_range(const std::string& name, int cid){
}

__inline__
void pop_range(const std::string& name){
}

__inline__
void pop_range(){
}

#endif
