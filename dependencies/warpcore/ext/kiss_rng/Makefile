CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -Wall -Wextra -fopenmp
NVCCGENCODE = -arch=sm_35
NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) --expt-extended-lambda -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

INC := $(wildcard $(INCDIR)/*)

all: example

example: example.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude example.cu -o example.out

debug: NVCCFLAGS += -g -O0 -Xptxas -v -UNDEBUG -D_DEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v
profile: all

clean:
	rm example.out
