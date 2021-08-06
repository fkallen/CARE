
CXX=g++
CUDACC=nvcc
HOSTLINKER=g++

CUB_INCDIR = ./dependencies/cub-cuda-11.2
THRUST_INCDIR = ./dependencies/thrust-cuda-11.2
WARPCORE_INCDIR = ./dependencies/warpcore/include


WARPCORE_FLAGS = -DCARE_HAS_WARPCORE -I$(WARPCORE_INCDIR)



CXXFLAGS = 

COMPILER_WARNINGS = -Wall -Wextra 
COMPILER_DISABLED_WARNING = -Wno-terminate -Wdeprecated-copy

CFLAGS_BASIC = $(COMPILER_WARNINGS) $(COMPILER_DISABLED_WARNING) -fopenmp -Iinclude -O3 -march=native -I$(THRUST_INCDIR)
CFLAGS_DEBUG_BASIC = $(COMPILER_WARNINGS) $(COMPILER_DISABLED_WARNING) -fopenmp -g -Iinclude -O0 -march=native -I$(THRUST_INCDIR)

CFLAGS_CPU = $(CFLAGS_BASIC) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CFLAGS_CPU_DEBUG = $(CFLAGS_DEBUG_BASIC) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP

NVCCFLAGS = -x cu -lineinfo -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) -I$(CUB_INCDIR) $(WARPCORE_FLAGS)
NVCCFLAGS_DEBUG = -x cu -rdc=true --expt-extended-lambda --expt-relaxed-constexpr -ccbin $(CXX) -I$(CUB_INCDIR) $(WARPCORE_FLAGS)

# This could be modified to compile only for a single architecture to reduce compilation time
CUDA_ARCH = -gencode=arch=compute_86,code=sm_86 
#\
#		-gencode=arch=compute_70,code=sm_70 \
#		-gencode=arch=compute_80,code=sm_80 \
 # 		-gencode=arch=compute_80,code=compute_80

LDFLAGSGPU = -lpthread -lgomp -lstdc++fs -lnvToolsExt -lz -ldl
LDFLAGSCPU = -lpthread -lgomp -lstdc++fs -lz -ldl

#sources for correct_cpu
SOURCES_CORRECT_CPU = \
    src/args.cpp \
    src/correct_cpu.cpp \
    src/correctionresultprocessing.cpp \
    src/cpu_alignment.cpp \
    src/cpuminhasherconstruction.cpp \
    src/dispatch_care_correct_cpu.cpp \
    src/main_correct_cpu.cpp \
    src/msa.cpp \
    src/readlibraryio.cpp \
    src/threadpool.cpp

#sources for correct_gpu
SOURCES_CORRECT_GPU = \
    src/args.cpp \
    src/readlibraryio.cpp \
    src/threadpool.cpp \
    src/gpu/alignmentkernels.cu \
    src/gpu/correct_gpu.cu \
    src/gpu/correctionkernels.cu \
    src/gpu/correctionresultprocessing_gpu.cu \
    src/gpu/dispatch_care_correct_gpu.cu \
    src/gpu/distributedreadstorage.cu \
    src/gpu/fakegpuminhasherconstruction.cu \
    src/gpu/gpucorrectorkernels.cu \
    src/gpu/gpuminhasherconstruction.cu \
    src/gpu/main_correct_gpu.cu \
    src/gpu/minhashingkernels.cu \
    src/gpu/msakernels.cu \
    src/gpu/multigpuminhasherconstruction.cu \
    src/gpu/sequenceconversionkernels.cu \
    src/gpu/singlegpuminhasherconstruction.cu

#sources for extend_cpu
SOURCES_EXTEND_CPU = \
    src/args.cpp \
    src/cpu_alignment.cpp \
    src/cpuminhasherconstruction.cpp \
    src/dispatch_care_extend_cpu.cpp \
    src/extensionresultprocessing.cpp \
    src/main_extend_cpu.cpp \
    src/msa.cpp \
    src/readextension.cpp \
    src/readlibraryio.cpp \
    src/threadpool.cpp

#sources for correct_gpu
SOURCES_EXTEND_GPU = \
    src/args.cpp \
    src/readlibraryio.cpp \
    src/threadpool.cpp \
    src/gpu/alignmentkernels.cu \
    src/gpu/dispatch_care_extend_gpu.cu \
    src/gpu/distributedreadstorage.cu \
    src/gpu/extensionresultprocessing_gpu.cu \
    src/gpu/fakegpuminhasherconstruction.cu \
    src/gpu/gpuminhasherconstruction.cu \
    src/gpu/main_extend_gpu.cu \
    src/gpu/minhashingkernels.cu \
    src/gpu/msakernels.cu \
    src/gpu/multigpuminhasherconstruction.cu \
    src/gpu/readextension_gpu.cu \
    src/gpu/sequenceconversionkernels.cu \
    src/gpu/singlegpuminhasherconstruction.cu


EXECUTABLE_CORRECT_CPU = care-cpu
EXECUTABLE_CORRECT_GPU = care-gpu
EXECUTABLE_EXTEND_CPU = care-extend-cpu
EXECUTABLE_EXTEND_GPU = care-extend-gpu

BUILDDIR_CORRECT_CPU = build_correct_cpu
BUILDDIR_CORRECT_GPU = build_correct_gpu
BUILDDIR_EXTEND_CPU = build_extend_cpu
BUILDDIR_EXTEND_GPU = build_extend_gpu

SOURCES_CORRECT_CPU_NODIR = $(notdir $(SOURCES_CORRECT_CPU))
SOURCES_CORRECT_GPU_NODIR = $(notdir $(SOURCES_CORRECT_GPU))
SOURCES_EXTEND_CPU_NODIR = $(notdir $(SOURCES_EXTEND_CPU))
SOURCES_EXTEND_GPU_NODIR = $(notdir $(SOURCES_EXTEND_GPU))

OBJECTS_CORRECT_CPU_NODIR = $(SOURCES_CORRECT_CPU_NODIR:%.cpp=%.o)
OBJECTS_CORRECT_GPU_NODIR_TMP = $(SOURCES_CORRECT_GPU_NODIR:%.cpp=%.o)
OBJECTS_CORRECT_GPU_NODIR = $(OBJECTS_CORRECT_GPU_NODIR_TMP:%.cu=%.o)
OBJECTS_EXTEND_CPU_NODIR = $(SOURCES_EXTEND_CPU_NODIR:%.cpp=%.o)
OBJECTS_EXTEND_GPU_NODIR_TMP = $(SOURCES_EXTEND_GPU_NODIR:%.cpp=%.o)
OBJECTS_EXTEND_GPU_NODIR = $(OBJECTS_EXTEND_GPU_NODIR_TMP:%.cu=%.o)

OBJECTS_CORRECT_CPU = $(OBJECTS_CORRECT_CPU_NODIR:%=$(BUILDDIR_CORRECT_CPU)/%)
OBJECTS_CORRECT_GPU = $(OBJECTS_CORRECT_GPU_NODIR:%=$(BUILDDIR_CORRECT_GPU)/%)
OBJECTS_EXTEND_CPU = $(OBJECTS_EXTEND_CPU_NODIR:%=$(BUILDDIR_EXTEND_CPU)/%)
OBJECTS_EXTEND_GPU = $(OBJECTS_EXTEND_GPU_NODIR:%=$(BUILDDIR_EXTEND_GPU)/%)



correct_cpu_release:
	@$(MAKE) correct_cpu_release_dummy DIR=$(BUILDDIR_CORRECT_CPU) CXXFLAGS="-std=c++14"

correct_gpu_release:
	@$(MAKE) correct_gpu_release_dummy DIR=$(BUILDDIR_CORRECT_GPU) CXXFLAGS="-std=c++14"

extend_cpu_release:
	@$(MAKE) extend_cpu_release_dummy DIR=$(BUILDDIR_EXTEND_CPU) CXXFLAGS="-std=c++17"

extend_gpu_release:
	@$(MAKE) extend_gpu_release_dummy DIR=$(BUILDDIR_EXTEND_GPU) CXXFLAGS="-std=c++17"




correct_cpu_release_dummy: $(BUILDDIR_CORRECT_CPU) $(OBJECTS_CORRECT_CPU) 
	@echo Linking $(EXECUTABLE_CORRECT_CPU)
	@$(HOSTLINKER) $(OBJECTS_CORRECT_CPU) $(LDFLAGSCPU) -o $(EXECUTABLE_CORRECT_CPU)
	@echo Linked $(EXECUTABLE_CORRECT_CPU)

correct_gpu_release_dummy: $(BUILDDIR_CORRECT_GPU) $(OBJECTS_CORRECT_GPU) 
	@echo Linking $(EXECUTABLE_CORRECT_GPU)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_CORRECT_GPU) $(LDFLAGSGPU) -o $(EXECUTABLE_CORRECT_GPU)
	@echo Linked $(EXECUTABLE_CORRECT_GPU)

extend_cpu_release_dummy: $(BUILDDIR_EXTEND_CPU) $(OBJECTS_EXTEND_CPU) 
	@echo Linking $(EXECUTABLE_EXTEND_CPU)
	@$(HOSTLINKER) $(OBJECTS_EXTEND_CPU) $(LDFLAGSCPU) -o $(EXECUTABLE_EXTEND_CPU)
	@echo Linked $(EXECUTABLE_EXTEND_CPU)

extend_gpu_release_dummy: $(BUILDDIR_EXTEND_GPU) $(OBJECTS_EXTEND_GPU) 
	@echo Linking $(EXECUTABLE_EXTEND_GPU)
	@$(CUDACC) $(CUDA_ARCH) $(OBJECTS_EXTEND_GPU) $(LDFLAGSGPU) -o $(EXECUTABLE_EXTEND_GPU)
	@echo Linked $(EXECUTABLE_EXTEND_GPU)


COMPILE = @echo "Compiling $< to $@" ; $(CXX) $(CXXFLAGS) $(CFLAGS_CPU) -c $< -o $@
CUDA_COMPILE = @echo "Compiling $< to $@" ; $(CUDACC) $(CUDA_ARCH) $(CXXFLAGS) $(NVCCFLAGS) -Xcompiler "$(CFLAGS_BASIC)" -c $< -o $@



.PHONY: cpu gpu extendcpu extendgpu clean
cpu: correct_cpu_release
gpu: correct_gpu_release
extendcpu: extend_cpu_release
extendgpu: extend_gpu_release

clean : 
	@rm -rf build_*
	@rm -f $(EXECUTABLE_CORRECT_CPU) 
	@rm -f $(EXECUTABLE_CORRECT_GPU)
	@rm -f $(EXECUTABLE_EXTEND_CPU)
	@rm -f $(EXECUTABLE_EXTEND_GPU)

$(DIR):
	mkdir $(DIR)

$(DIR)/args.o : src/args.cpp
	$(COMPILE)

$(DIR)/correct_cpu.o : src/correct_cpu.cpp
	$(COMPILE)

$(DIR)/correctionresultprocessing.o : src/correctionresultprocessing.cpp
	$(COMPILE)

$(DIR)/cpu_alignment.o : src/cpu_alignment.cpp
	$(COMPILE)

$(DIR)/cpuminhasherconstruction.o : src/cpuminhasherconstruction.cpp
	$(COMPILE)

$(DIR)/dispatch_care_correct_cpu.o : src/dispatch_care_correct_cpu.cpp
	$(COMPILE)

$(DIR)/dispatch_care_extend_cpu.o : src/dispatch_care_extend_cpu.cpp
	$(COMPILE)

$(DIR)/extensionresultprocessing.o : src/extensionresultprocessing.cpp
	$(COMPILE)

$(DIR)/main_correct_cpu.o : src/main_correct_cpu.cpp
	$(COMPILE)

$(DIR)/main_extend_cpu.o : src/main_extend_cpu.cpp
	$(COMPILE)

$(DIR)/msa.o : src/msa.cpp
	$(COMPILE)

$(DIR)/readextension.o : src/readextension.cpp
	$(COMPILE)

$(DIR)/readlibraryio.o : src/readlibraryio.cpp
	$(COMPILE)

$(DIR)/semi_global_alignment.o : src/semi_global_alignment.cpp
	$(COMPILE)

$(DIR)/threadpool.o : src/threadpool.cpp
	$(COMPILE)

$(DIR)/alignmentkernels.o : src/gpu/alignmentkernels.cu
	$(CUDA_COMPILE)

$(DIR)/correct_gpu.o : src/gpu/correct_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/correctionkernels.o : src/gpu/correctionkernels.cu
	$(CUDA_COMPILE)

$(DIR)/correctionresultprocessing_gpu.o : src/gpu/correctionresultprocessing_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/dispatch_care_correct_gpu.o : src/gpu/dispatch_care_correct_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/dispatch_care_extend_gpu.o : src/gpu/dispatch_care_extend_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/distributedreadstorage.o : src/gpu/distributedreadstorage.cu
	$(CUDA_COMPILE)

$(DIR)/extensionresultprocessing_gpu.o : src/gpu/extensionresultprocessing_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/fakegpuminhasherconstruction.o : src/gpu/fakegpuminhasherconstruction.cu
	$(CUDA_COMPILE)

$(DIR)/gpucorrectorkernels.o : src/gpu/gpucorrectorkernels.cu
	$(CUDA_COMPILE)

$(DIR)/gpuminhasherconstruction.o : src/gpu/gpuminhasherconstruction.cu
	$(CUDA_COMPILE)

$(DIR)/main_correct_gpu.o : src/gpu/main_correct_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/main_extend_gpu.o : src/gpu/main_extend_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/minhashingkernels.o : src/gpu/minhashingkernels.cu
	$(CUDA_COMPILE)

$(DIR)/msakernels.o : src/gpu/msakernels.cu
	$(CUDA_COMPILE)

$(DIR)/multigpuminhasherconstruction.o : src/gpu/multigpuminhasherconstruction.cu
	$(CUDA_COMPILE)

$(DIR)/readextension_gpu.o : src/gpu/readextension_gpu.cu
	$(CUDA_COMPILE)

$(DIR)/sequenceconversionkernels.o : src/gpu/sequenceconversionkernels.cu
	$(CUDA_COMPILE)

$(DIR)/singlegpuminhasherconstruction.o : src/gpu/singlegpuminhasherconstruction.cu
	$(CUDA_COMPILE)


