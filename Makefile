CXX=g++
CUDACC=nvcc
HOSTLINKER=g++

CXXFLAGS = -std=c++14
CFLAGS = -Wall -fopenmp -g -Iinclude -O3 -march=native
CFLAGS_DEBUG = -Wall -fopenmp -g -Iinclude

CUB_INCLUDE = -I/home/fekallen/cub-1.8.0

NVCCFLAGS = -x cu -lineinfo -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)
NVCCFLAGS_DEBUG = -G -x cu -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)

#TODO CUDA_PATH =

CUDA_ARCH = -gencode=arch=compute_61,code=sm_61



LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt -ldl -lz -lpython2.7
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs -ldl -lz -lpython2.7


# sources which are used by both cpu version and gpu version
SOURCES_CPU_AND_GPU_ = $(wildcard src/*.cpp)
SOURCES_CPU_AND_GPU = $(filter-out src/cpugpuproxy_cpu.cpp src/dispatch_care_cpu.cpp src/minhasher_transform.cpp,$(SOURCES_CPU_AND_GPU_))

# sources which are used by gpu version exclusively
SOURCES_ONLY_GPU = $(wildcard src/gpu/*.cu)
#SOURCES_ONLY_GPU = $(filter-out src/gpu/correct_gpu.cu, $(SOURCES_ONLY_GPU_))

# sources which are used by cpu version exclusively
SOURCES_ONLY_CPU = src/cpugpuproxy_cpu.cpp src/dispatch_care_cpu.cpp src/minhasher_transform.cpp


OBJECTS_CPU_AND_GPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES_CPU_AND_GPU))
OBJECTS_CPU_AND_GPU_DEBUG = $(patsubst src/%.cpp, buildcpu/%.dbg.o, $(SOURCES_CPU_AND_GPU))

OBJECTS_ONLY_GPU = $(patsubst src/gpu/%.cu, buildgpu/%.o, $(SOURCES_ONLY_GPU))
OBJECTS_ONLY_GPU_DEBUG = $(patsubst src/gpu/%.cu, buildgpu/%.dbg.o, $(SOURCES_ONLY_GPU))

OBJECTS_ONLY_CPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES_ONLY_CPU))
OBJECTS_ONLY_CPU_DEBUG = $(patsubst src/%.cpp, buildcpu/%.dbg.o, $(SOURCES_ONLY_CPU))


SOURCES_FORESTS = $(wildcard src/forests/*.cpp)
OBJECTS_FORESTS = $(patsubst src/forests/%.cpp, forests/%.so, $(SOURCES_FORESTS))
OBJECTS_FORESTS_DEBUG = $(patsubst src/forests/%.cpp, forests/%.dbg.so, $(SOURCES_FORESTS))

#$(info $SOURCES_CPU_AND_GPU is [${SOURCES_CPU_AND_GPU}])
#$(info $SOURCES_ONLY_GPU is [${SOURCES_ONLY_GPU}])
#$(info $SORCES_ONLY_CPU is [${SORCES_ONLY_CPU}])

#$(info $$OBJECTS_FORESTS is [${OBJECTS_FORESTS}])
#$(info $$OBJECTS_FORESTS_DEBUG is [${OBJECTS_FORESTS_DEBUG}])


GPU_VERSION = errorcorrector_gpu
CPU_VERSION = errorcorrector_cpu
GPU_VERSION_DEBUG = errorcorrector_gpu_debug
CPU_VERSION_DEBUG = errorcorrector_cpu_debug


all: cpu

cpu:	$(CPU_VERSION)
gpu:	$(GPU_VERSION)
cpud:	$(CPU_VERSION_DEBUG)
gpud:	$(GPU_VERSION_DEBUG)

forests:	$(OBJECTS_FORESTS) $(OBJECTS_FORESTS_DEBUG)

$(GPU_VERSION) : $(OBJECTS_ONLY_GPU) $(OBJECTS_CPU_AND_GPU)
	@echo Linking $(GPU_VERSION)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_ONLY_GPU) $(OBJECTS_CPU_AND_GPU) $(LDFLAGSGPU) -o $(GPU_VERSION)

$(CPU_VERSION) : $(OBJECTS_ONLY_CPU) $(OBJECTS_CPU_AND_GPU)
	@echo Linking $(CPU_VERSION)
	@$(HOSTLINKER) $(OBJECTS_ONLY_CPU) $(OBJECTS_CPU_AND_GPU) $(LDFLAGSCPU) -o $(CPU_VERSION)

$(GPU_VERSION_DEBUG) : $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
	@echo Linking $(GPU_VERSION_DEBUG)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG) $(LDFLAGSGPU) -o $(GPU_VERSION_DEBUG)

$(CPU_VERSION_DEBUG) : $(OBJECTS_ONLY_CPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
	@echo Linking $(CPU_VERSION_DEBUG)
	@$(HOSTLINKER) $(OBJECTS_ONLY_CPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG) $(LDFLAGSCPU) -o $(CPU_VERSION_DEBUG)

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

forests/%.so : src/forests/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS) -shared -fPIC $< -o $@

forests/%.dbg.so : src/forests/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS_DEBUG) -shared -fPIC $< -o $@

minhashertest:
	@echo Building minhashertest
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" tests/minhashertest/main.cpp src/sequencefileio.cpp $(LDFLAGSGPU) -o tests/minhashertest/main

alignmenttest:
	@echo Building alignmenttest
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS_DEBUG) -Xcompiler "$(CFLAGS_DEBUG)" tests/alignmenttest/main.cpp src/gpu/kernels.cu  $(LDFLAGSGPU) -o tests/alignmenttest/main

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
	@mkdir -p debugbuildcpu
	@mkdir -p debugbuildgpu
	@mkdir -p forests

.PHONY: minhashertest

.PHONY: makedirs

-PHONY: forests
