CXX=g++
CUDACC=nvcc
HOSTLINKER=g++


CUB_INCLUDE = -I/home/fekallen/cub-1.8.0
THRUST_INCLUDE = -I/usr/local/cuda/include/

CXXFLAGS = -std=c++14
CFLAGS = -Wall -fopenmp -g -Iinclude -O3 -march=native $(THRUST_INCLUDE)
CFLAGS_DEBUG = -Wall -fopenmp -g -O0 -Iinclude $(THRUST_INCLUDE)


NVCCFLAGS = -x cu -lineinfo -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)
NVCCFLAGS_DEBUG = -G -x cu -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)

#TODO CUDA_PATH =

CUDA_ARCH = -gencode=arch=compute_61,code=sm_61



LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt -lz 
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs -lz 


# sources which are used by both cpu version and gpu version
SOURCES_CPU_AND_GPU_ = $(wildcard src/*.cpp)
#SOURCES_CPU_AND_GPU = $(filter-out src/cpugpuproxy_cpu.cpp src/dispatch_care_cpu.cpp src/minhasher_transform.cpp,$(SOURCES_CPU_AND_GPU_))
SOURCES_CPU_AND_GPU = $(filter-out src/correct_cpu.cpp src/cpugpuproxy_cpu.cpp src/dispatch_care_cpu.cpp,$(SOURCES_CPU_AND_GPU_))

# sources which are used by gpu version exclusively
SOURCES_ONLY_GPU = $(wildcard src/gpu/*.cu)
#SOURCES_ONLY_GPU = $(filter-out src/gpu/correct_gpu.cu, $(SOURCES_ONLY_GPU_))

# sources which are used by cpu version exclusively
#SOURCES_ONLY_CPU = src/cpugpuproxy_cpu.cpp src/dispatch_care_cpu.cpp src/minhasher_transform.cpp
SOURCES_ONLY_CPU = src/correct_cpu.cpp src/cpugpuproxy_cpu.cpp src/dispatch_care_cpu.cpp


OBJECTS_CPU_AND_GPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES_CPU_AND_GPU))
OBJECTS_CPU_AND_GPU_DEBUG = $(patsubst src/%.cpp, buildcpu/%.dbg.o, $(SOURCES_CPU_AND_GPU))

OBJECTS_ONLY_GPU = $(patsubst src/gpu/%.cu, buildgpu/%.o, $(SOURCES_ONLY_GPU))
OBJECTS_ONLY_GPU_DEBUG = $(patsubst src/gpu/%.cu, buildgpu/%.dbg.o, $(SOURCES_ONLY_GPU))

OBJECTS_ONLY_CPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES_ONLY_CPU))
OBJECTS_ONLY_CPU_DEBUG = $(patsubst src/%.cpp, buildcpu/%.dbg.o, $(SOURCES_ONLY_CPU))



#$(info $SOURCES_CPU_AND_GPU is [${SOURCES_CPU_AND_GPU}])
#$(info $SOURCES_ONLY_GPU is [${SOURCES_ONLY_GPU}])
#$(info $SORCES_ONLY_CPU is [${SORCES_ONLY_CPU}])


GPU_VERSION = errorcorrector_gpu
CPU_VERSION = errorcorrector_cpu
GPU_VERSION_DEBUG = errorcorrector_gpu_debug
CPU_VERSION_DEBUG = errorcorrector_cpu_debug


all: cpu

cpu:	$(CPU_VERSION)
gpu:	$(GPU_VERSION)
cpud:	$(CPU_VERSION_DEBUG)
gpud:	$(GPU_VERSION_DEBUG)


$(GPU_VERSION) : $(OBJECTS_ONLY_GPU) $(OBJECTS_CPU_AND_GPU)
	@echo Linking $(GPU_VERSION)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_ONLY_GPU) $(OBJECTS_CPU_AND_GPU) $(LDFLAGSGPU) -o $(GPU_VERSION)
	@echo Linked $(GPU_VERSION)

$(CPU_VERSION) : $(OBJECTS_ONLY_CPU) $(OBJECTS_CPU_AND_GPU)
	@echo Linking $(CPU_VERSION)
	@$(HOSTLINKER) $(OBJECTS_ONLY_CPU) $(OBJECTS_CPU_AND_GPU) $(LDFLAGSCPU) -o $(CPU_VERSION)
	@echo Linked $(CPU_VERSION)

$(GPU_VERSION_DEBUG) : $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
	@echo Linking $(GPU_VERSION_DEBUG)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG) $(LDFLAGSGPU) -o $(GPU_VERSION_DEBUG)
	@echo Linked $(GPU_VERSION_DEBUG)

$(CPU_VERSION_DEBUG) : $(OBJECTS_ONLY_CPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
	@echo Linking $(CPU_VERSION_DEBUG)
	@$(HOSTLINKER) $(OBJECTS_ONLY_CPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG) $(LDFLAGSCPU) -o $(CPU_VERSION_DEBUG)
	@echo Linked $(CPU_VERSION_DEBUG)

buildcpu/%.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

buildcpu/%.dbg.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS_DEBUG) -c $< -o $@

buildgpu/%.o : src/gpu/%.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

buildgpu/%.dbg.o : src/gpu/%.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS_DEBUG) -Xcompiler "$(CFLAGS_DEBUG)" -c $< -o $@



minhashertest:
	@echo Building minhashertest
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" tests/minhashertest/main.cpp src/sequencefileio.cpp $(LDFLAGSGPU) -o tests/minhashertest/main

alignmenttest:
	@echo Building alignmenttest
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS_DEBUG) -Xcompiler "$(CFLAGS_DEBUG)" tests/alignmenttest/main.cpp src/gpu/kernels.cu  $(LDFLAGSGPU) -o tests/alignmenttest/main

makeresultfile: buildcpu/sequencefileio.o makeresultfile/makeresultfile.cpp
	g++ $(CXXFLAGS) $(CFLAGS) makeresultfile/makeresultfile.cpp buildcpu/sequencefileio.o -lz -lstdc++fs -o makeresultfile/makeresultfile

clean:
	@rm -f $(GPU_VERSION) $(CPU_VERSION) $(GPU_VERSION_DEBUG) $(CPU_VERSION_DEBUG)\
			$(OBJECTS_CPU_AND_GPU) $(OBJECTS_ONLY_GPU) $(OBJECTS_ONLY_CPU) \
			$(OBJECTS_CPU_AND_GPU_DEBUG) $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_ONLY_CPU_DEBUG)
cleancpu:
	@rm -f $(CPU_VERSION) $(OBJECTS_ONLY_CPU) $(OBJECTS_CPU_AND_GPU)
cleangpu:
	@rm -f $(GPU_VERSION) $(OBJECTS_ONLY_GPU) $(OBJECTS_CPU_AND_GPU)
cleancpud:
	@rm -f $(CPU_VERSION_DEBUG) $(OBJECTS_ONLY_CPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
cleangpud:
	@rm -f $(GPU_VERSION_DEBUG) $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
makedir:
	@mkdir -p buildcpu
	@mkdir -p buildgpu
	@mkdir -p forests

.PHONY: minhashertest

.PHONY: makedirs

.PHONY: makeresultfile
