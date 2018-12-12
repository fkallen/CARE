CXX=g++
CUDACC=nvcc
HOSTLINKER=g++

CXXFLAGS = -std=c++14
CFLAGS = -Wall -fopenmp -g -Iinclude -O3
CFLAGS_DEBUG = -Wall -fopenmp -g -Iinclude

CUB_INCLUDE = -I/home/fekallen/cub-1.8.0

NVCCFLAGS = -x cu -lineinfo -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)
NVCCFLAGS_DEBUG = -G -x cu -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)

#TODO CUDA_PATH =

CUDA_ARCH = -gencode=arch=compute_61,code=sm_61



LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs


SOURCES_CPU = $(wildcard src/*.cpp)
SOURCES_GPU = src/gpu/kernels.cu src/gpu/qualityscoreweights.cu src/care.cpp

OBJECTS_CPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES_CPU))
OBJECTS_CPU_DEBUG = $(patsubst src/%.cpp, buildcpu/%.dbg.o, $(SOURCES_CPU))
OBJECTS_GPU_ = $(patsubst src/%.cpp, buildgpu/%.o, $(SOURCES_GPU))
OBJECTS_GPU = $(patsubst src/gpu/%.cu, buildgpu/%.o, $(OBJECTS_GPU_))
OBJECTS_GPU_DEBUG = $(patsubst buildgpu/%.o, buildgpu/%.dbg.o, $(OBJECTS_GPU))

OBJECTS_CPU_AND_GPU = $(filter-out buildcpu/care.o,$(OBJECTS_CPU))
OBJECTS_CPU_AND_GPU_DEBUG = $(filter-out buildcpu/care.dbg.o,$(OBJECTS_CPU_DEBUG))

#$(info $$SOURCES_CPU is [${SOURCES_CPU}])
#$(info $$SOURCES_GPU is [${SOURCES_GPU}])
#$(info $$OBJECTS_GPU is [${OBJECTS_GPU}])


PATH_CORRECTOR=$(shell pwd)

INC_CORRECTOR=$(PATH_CORRECTOR)/include

GPU_VERSION = errorcorrector_gpu
CPU_VERSION = errorcorrector_cpu
GPU_VERSION_DEBUG = errorcorrector_gpu_debug
CPU_VERSION_DEBUG = errorcorrector_cpu_debug


all: cpu

cpu:	$(CPU_VERSION)
gpu:	$(GPU_VERSION)
cpud:	$(CPU_VERSION_DEBUG)
gpud:	$(GPU_VERSION_DEBUG)



$(GPU_VERSION) : $(OBJECTS_GPU) $(OBJECTS_CPU_AND_GPU)
	@echo Linking $(GPU_VERSION) from $(OBJECTS_GPU) $(OBJECTS_CPU_AND_GPU)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_GPU) $(OBJECTS_CPU_AND_GPU) $(LDFLAGSGPU) -o $(GPU_VERSION)

$(CPU_VERSION) : $(OBJECTS_CPU)
	@echo Linking $(CPU_VERSION)
	@$(HOSTLINKER) $(OBJECTS_CPU) $(LDFLAGSCPU) -o $(CPU_VERSION)

$(GPU_VERSION_DEBUG) : $(OBJECTS_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
	@echo Linking $(GPU_VERSION_DEBUG)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG) $(LDFLAGSGPU) -o $(GPU_VERSION_DEBUG)

$(CPU_VERSION_DEBUG) : $(OBJECTS_CPU_DEBUG)
	@echo Linking $(CPU_VERSION_DEBUG)
	@$(HOSTLINKER) $(OBJECTS_CPU_DEBUG) $(LDFLAGSCPU) -o $(CPU_VERSION_DEBUG)

buildcpu/%.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

buildcpu/%.dbg.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS_DEBUG) -c $< -o $@

buildgpu/kernels.o : src/gpu/kernels.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

buildgpu/qualityscoreweights.o : src/gpu/qualityscoreweights.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

buildgpu/care.o : src/care.cpp | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

buildgpu/kernels.dbg.o : src/gpu/kernels.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

buildgpu/qualityscoreweights.dbg.o : src/gpu/qualityscoreweights.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

buildgpu/care.dbg.o : src/care.cpp | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

minhashertest:
	@echo Building minhashertest
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" tests/minhashertest/main.cpp src/sequencefileio.cpp $(LDFLAGSGPU) -o tests/minhashertest/main

clean:
	@rm -f $(GPU_VERSION) $(CPU_VERSION) $(GPU_VERSION_DEBUG) $(CPU_VERSION_DEBUG) $(OBJECTS_GPU) $(OBJECTS_CPU) $(OBJECTS_GPU_DEBUG) $(OBJECTS_CPU_DEBUG)
cleancpu:
	@rm -f $(CPU_VERSION) $(OBJECTS_CPU)
cleangpu:
	@rm -f $(GPU_VERSION) $(OBJECTS_GPU)
cleancpud:
	@rm -f $(CPU_VERSION_DEBUG) $(OBJECTS_CPU_DEBUG)
cleangpud:
	@rm -f $(GPU_VERSION_DEBUG) $(OBJECTS_GPU_DEBUG)
makedir:
	@mkdir -p buildcpu
	@mkdir -p buildgpu
	@mkdir -p debugbuildcpu
	@mkdir -p debugbuildgpu

.PHONY: minhashertest

.PHONY: makedirs
