
# Location of the CUDA Toolkit
CUDA_PATH ?= "/gpfs/loomis/apps/avx/software/CUDA/10.1.105"

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   :=
CCFLAGS     :=
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS += -g
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/samples/common/inc
LIBRARIES :=

################################################################################

# libNVRTC specific libraries
CUDA_SEARCH_PATH ?= $(CUDA_PATH)/lib64/stubs
CUDA_SEARCH_PATH += $(CUDA_PATH)/targets/x86_64-linux/lib/stubs

CUDALIB ?= $(shell find -L $(CUDA_SEARCH_PATH) -maxdepth 1 -name libcuda.so 2> /dev/null)
ifeq ("$(CUDALIB)","")
  $(info >>> WARNING - libcuda.so not found, CUDA Driver is not installed.  Please re-install the driver. <<<)
  SAMPLE_ENABLED := 0
else
  CUDALIB := $(shell echo $(CUDALIB) | sed "s/ .*//" | sed "s/\/libcuda.so//" )
  LIBRARIES += -L$(CUDALIB) -lcuda
endif

INCLUDES += -I$(CUDA_PATH)/include

LIBRARIES += -lnvrtc -lcudart

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

# Target rules
all: build

build: vectorAdd_nvrtc

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

vectorAdd.o:vectorAdd.cpp
	$(EXEC) $(HOST_COMPILER) $(INCLUDES) $(CCFLAGS) $(EXTRA_CCFLAGS) -o $@ -c $<

vectorAdd_nvrtc: vectorAdd.o
	$(EXEC) $(HOST_COMPILER) $(LDFLAGS) -o $@ $+ $(LIBRARIES)

run: build
	$(EXEC) ./vectorAdd_nvrtc

clean:
	rm -f vectorAdd_nvrtc vectorAdd.o

clobber: clean
