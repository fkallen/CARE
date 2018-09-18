CXX=g++
CUDACC=nvcc
HOSTLINKER=g++

CXXFLAGS = -std=c++14
CFLAGS = -Wall -fopenmp -g -Iinc -O3
CFLAGS_DEBUG = -Wall -fopenmp -g -Iinc

CUB_INCLUDE = -I/home/fekallen/cub-1.8.0

NVCCFLAGS = -x cu -lineinfo -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)
NVCCFLAGS_DEBUG = -G -x cu -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) $(CUB_INCLUDE)

#TODO CUDA_PATH =

CUDA_ARCH = -gencode=arch=compute_61,code=sm_61



LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs

SOURCES = $(wildcard src/*.cpp)
OBJECTS_CPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES))
OBJECTS_GPU = $(patsubst src/%.cpp, buildgpu/%.o, $(SOURCES)) buildgpu/kernels.o
OBJECTS_CPU_DEBUG = $(patsubst src/%.cpp, debugbuildcpu/%.o, $(SOURCES))
OBJECTS_GPU_DEBUG = $(patsubst src/%.cpp, debugbuildgpu/%.o, $(SOURCES)) debugbuildgpu/kernels.o

PATH_CORRECTOR=$(shell pwd)

INC_CORRECTOR=$(PATH_CORRECTOR)/inc

GPU_VERSION = errorcorrector_gpu
CPU_VERSION = errorcorrector_cpu
GPU_VERSION_DEBUG = errorcorrector_gpu_debug
CPU_VERSION_DEBUG = errorcorrector_cpu_debug


all: cpu

cpu:	$(CPU_VERSION)
gpu:	$(GPU_VERSION)
cpud:	$(CPU_VERSION_DEBUG)
gpud:	$(GPU_VERSION_DEBUG)



$(GPU_VERSION) : $(OBJECTS_GPU)
	@echo Linking $(GPU_VERSION)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_GPU) $(LDFLAGSGPU) -o $(GPU_VERSION)

$(CPU_VERSION) : $(OBJECTS_CPU)
	@echo Linking $(CPU_VERSION)
	@$(HOSTLINKER) $(OBJECTS_CPU) $(LDFLAGSCPU) -o $(CPU_VERSION)

$(GPU_VERSION_DEBUG) : $(OBJECTS_GPU_DEBUG)
	@echo Linking $(GPU_VERSION_DEBUG)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_GPU_DEBUG) $(LDFLAGSGPU) -o $(GPU_VERSION_DEBUG)

$(CPU_VERSION_DEBUG) : $(OBJECTS_CPU_DEBUG)
	@echo Linking $(CPU_VERSION_DEBUG)
	@$(HOSTLINKER) $(OBJECTS_CPU_DEBUG) $(LDFLAGSCPU) -o $(CPU_VERSION_DEBUG)

buildcpu/%.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS) -c $< -o $@

buildgpu/%.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

buildgpu/kernels.o : inc/gpu_only_path/kernels.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" -c $< -o $@

debugbuildcpu/%.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS_DEBUG) -c $< -o $@

debugbuildgpu/%.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS_DEBUG) -Xcompiler "$(CFLAGS_DEBUG)" -c $< -o $@

debugbuildgpu/kernels.o : inc/gpu_only_path/kernels.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS_DEBUG) -Xcompiler "$(CFLAGS_DEBUG)" -c $< -o $@

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
