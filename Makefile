
PREFIX=$(shell cat .PREFIX)
CUDA_DIR=$(shell cat .CUDA_DIR)
CUB_INCDIR=$(shell cat .CUB_INCDIR)
THRUST_INCDIR=$(shell cat .THRUST_INCDIR)

BUILD_WITH_WARPCORE = 1

ifeq ($(BUILD_WITH_WARPCORE), 1)
	WARPCORE_INCDIR = $(shell cat .WARPCORE_INCDIR)
	WARPCORE_INCLUDE_FLAGS = -I$(WARPCORE_INCDIR)
	WARPCORE_CFLAGS = -DCARE_HAS_WARPCORE
else
	WARPCORE_INCDIR = 
	WARPCORE_INCLUDE_FLAGS = 
	WARPCORE_CFLAGS = 
endif

CXX=g++
CUDACC=$(CUDA_DIR)/bin/nvcc
HOSTLINKER=g++

CXXFLAGS = -std=c++17

CFLAGS_BASIC = -Wall -fopenmp -g -Iinclude -O3 -march=native -I$(THRUST_INCDIR)
CFLAGS_DEBUG_BASIC = -Wall -fopenmp -g -Iinclude -O0 -march=native -I$(THRUST_INCDIR)

CFLAGS_CPU = $(CFLAGS_BASIC) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CFLAGS_CPU_DEBUG = $(CFLAGS_DEBUG_BASIC) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

NVCCFLAGS = -x cu -lineinfo -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) -I$(CUB_INCDIR) $(WARPCORE_INCLUDE_FLAGS) $(WARPCORE_CFLAGS)
NVCCFLAGS_DEBUG = -x cu -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) -I$(CUB_INCDIR) $(WARPCORE_INCLUDE_FLAGS) $(WARPCORE_CFLAGS)

# This could be modified to compile only for a single architecture to reduce compilation time
CUDA_ARCH = -gencode=arch=compute_61,code=sm_61 
#\
#		-gencode=arch=compute_70,code=sm_70 \
#		-gencode=arch=compute_80,code=sm_80 \
 # 		-gencode=arch=compute_80,code=compute_80

LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt -lz -ldl
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs -lz -ldl

# sources which are used by gpu version exclusively
SOURCES_ONLY_GPU = $(wildcard src/gpu/*.cu)

# sources which are used by cpu version exclusively
# src/correct_cpu.cpp 
SOURCES_ONLY_CPU = src/dispatch_care_cpu.cpp src/correctionresultprocessing.cpp src/extensionresultprocessing.cpp src/readextension.cpp

# sources which are used by both cpu version and gpu version
SOURCES_CPU_AND_GPU_ = $(wildcard src/*.cpp)
SOURCES_CPU_AND_GPU = $(filter-out $(SOURCES_ONLY_CPU), $(SOURCES_CPU_AND_GPU_))

# sources of ML forests
SOURCES_FORESTS = $(wildcard forests/*.cpp)


OBJECTS_CPU_AND_GPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES_CPU_AND_GPU))
OBJECTS_CPU_AND_GPU_DEBUG = $(patsubst src/%.cpp, buildcpu/%.dbg.o, $(SOURCES_CPU_AND_GPU))

OBJECTS_ONLY_GPU = $(patsubst src/gpu/%.cu, buildgpu/%.o, $(SOURCES_ONLY_GPU))
OBJECTS_ONLY_GPU_DEBUG = $(patsubst src/gpu/%.cu, buildgpu/%.dbg.o, $(SOURCES_ONLY_GPU))

OBJECTS_ONLY_CPU = $(patsubst src/%.cpp, buildcpu/%.o, $(SOURCES_ONLY_CPU))
OBJECTS_ONLY_CPU_DEBUG = $(patsubst src/%.cpp, buildcpu/%.dbg.o, $(SOURCES_ONLY_CPU))

OBJECTS_FORESTS = $(patsubst forests/%.cpp, forests/%.so, $(SOURCES_FORESTS))
OBJECTS_FORESTS_DEBUG = $(patsubst forests/%.cpp, forests/%.dbg.so, $(SOURCES_FORESTS))


GPU_VERSION = care-gpu
CPU_VERSION = care-cpu
GPU_VERSION_DEBUG = care-gpu-debug
CPU_VERSION_DEBUG = care-cpu-debug


all: cpu

cpu:	$(CPU_VERSION)
gpu:	$(GPU_VERSION)
cpud:	$(CPU_VERSION_DEBUG)
gpud:	$(GPU_VERSION_DEBUG)

forests:	$(OBJECTS_FORESTS) 
#$(OBJECTS_FORESTS_DEBUG)


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
	@$(CXX) $(CXXFLAGS) $(CFLAGS_CPU) -c $< -o $@

buildcpu/%.dbg.o : src/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS_CPU_DEBUG) -c $< -o $@

buildgpu/%.o : src/gpu/%.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS_BASIC)" -c $< -o $@

buildgpu/%.dbg.o : src/gpu/%.cu | makedir
	@echo Compiling $< to $@
	@$(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS_DEBUG) -Xcompiler "$(CFLAGS_DEBUG_BASIC)" -c $< -o $@


forests/%.so : forests/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS) -shared -fPIC $< -o $@

forests/%.dbg.so : forests/%.cpp | makedir
	@echo Compiling $< to $@
	@$(CXX) $(CXXFLAGS) $(CFLAGS_DEBUG) -shared -fPIC $< -o $@



install: 
	mkdir -p $(PREFIX)/bin
ifneq ("$(wildcard $(CPU_VERSION))","")
	cp $(CPU_VERSION) $(PREFIX)/bin/$(CPU_VERSION)
endif	
ifneq ("$(wildcard $(GPU_VERSION))","")
	cp $(GPU_VERSION) $(PREFIX)/bin/$(GPU_VERSION)
endif

clean:
	@rm -f $(GPU_VERSION) $(CPU_VERSION) $(GPU_VERSION_DEBUG) $(CPU_VERSION_DEBUG)\
			$(OBJECTS_CPU_AND_GPU) $(OBJECTS_ONLY_GPU) $(OBJECTS_ONLY_CPU) \
			$(OBJECTS_CPU_AND_GPU_DEBUG) $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_ONLY_CPU_DEBUG) \
			$(OBJECTS_FORESTS) $(OBJECTS_FORESTS_DEBUG)
cleancpu:
	@rm -f $(CPU_VERSION) $(OBJECTS_ONLY_CPU) $(OBJECTS_CPU_AND_GPU)
cleangpu:
	@rm -f $(GPU_VERSION) $(OBJECTS_ONLY_GPU) $(OBJECTS_CPU_AND_GPU)
cleancpud:
	@rm -f $(CPU_VERSION_DEBUG) $(OBJECTS_ONLY_CPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
cleangpud:
	@rm -f $(GPU_VERSION_DEBUG) $(OBJECTS_ONLY_GPU_DEBUG) $(OBJECTS_CPU_AND_GPU_DEBUG)
cleanforests:
	@rm -f $(OBJECTS_FORESTS) $(OBJECTS_FORESTS_DEBUG)

makedir:
	@mkdir -p buildcpu
	@mkdir -p buildgpu
	@mkdir -p forests

.PHONY: makedir
