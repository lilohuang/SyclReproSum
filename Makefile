# ============================================================
#  Ahrens-Demmel-Nguyen K-Fold Reproducible Summation - Makefile
# ============================================================
#
#  Targets:
#    make            - build example (repro_example)
#    make test       - build & run Google Test suite
#    make all        - build both example and test
#    make run        - build & run example
#    make test-validation - run hostile host/device validation tests
#    make clean      - remove binaries
#    make gtest      - build gtest from bundled source (once)
#
#  Configuration:
#    ENABLE_AMD      - include AMD HIP support (default: 1)
#    ENABLE_NVIDIA   - include NVIDIA CUDA support (default: 1)
#    ENABLE_SPIRV    - include SPIR-V CPU/GPU support (default: 1)
#    DPCPP_HOME      - path to DPC++ installation (default: ~/sycl_workspace)
#    SYCL_TARGETS    - explicit offload target override
#    ONEDPL_INC      - oneDPL include directory (auto-detected when possible)
#    CUDA_ARCH       - NVIDIA GPU architecture (auto-detected when possible)
#    AMD_GPU_ARCH    - AMDGPU architecture (auto-detected when possible)
#    AMD_LIBSPIRV    - AMD libspirv bitcode file (auto-detected)
# ============================================================

ENABLE_AMD    ?= 1
ENABLE_NVIDIA ?= 1
ENABLE_SPIRV  ?= 1

ifneq ($(filter $(ENABLE_AMD),0 1),$(ENABLE_AMD))
$(error ENABLE_AMD must be 0 or 1)
endif
ifneq ($(filter $(ENABLE_NVIDIA),0 1),$(ENABLE_NVIDIA))
$(error ENABLE_NVIDIA must be 0 or 1)
endif
ifneq ($(filter $(ENABLE_SPIRV),0 1),$(ENABLE_SPIRV))
$(error ENABLE_SPIRV must be 0 or 1)
endif

NVIDIA_TARGET = nvptx64-nvidia-cuda
SPIRV_TARGET  = spir64
AMD_TARGET    = amdgcn-amd-amdhsa

empty :=
space := $(empty) $(empty)
comma := ,

ENABLED_SYCL_TARGETS :=
ifeq ($(ENABLE_NVIDIA),1)
ENABLED_SYCL_TARGETS += $(NVIDIA_TARGET)
endif
ifeq ($(ENABLE_SPIRV),1)
ENABLED_SYCL_TARGETS += $(SPIRV_TARGET)
endif
ifeq ($(ENABLE_AMD),1)
ENABLED_SYCL_TARGETS += $(AMD_TARGET)
endif

ENABLED_SYCL_TARGETS := $(strip $(ENABLED_SYCL_TARGETS))

DPCPP_HOME   ?= $(HOME)/sycl_workspace
SYCL_TARGETS ?= $(subst $(space),$(comma),$(ENABLED_SYCL_TARGETS))
ONEDPL_INC   ?= $(firstword \
	$(wildcard /opt/intel/oneapi/dpl/latest/include) \
	$(wildcard $(DPCPP_HOME)/oneDPL*/include) \
	/opt/intel/oneapi/dpl/latest/include)

ifeq ($(strip $(SYCL_TARGETS)),)
$(error At least one SYCL target must be enabled)
endif

SYCL_TARGET_LIST     = $(subst $(comma), ,$(SYCL_TARGETS))
NVIDIA_TARGET_ENABLED = $(filter $(NVIDIA_TARGET),$(SYCL_TARGET_LIST))
SPIRV_TARGET_ENABLED = $(filter $(SPIRV_TARGET),$(SYCL_TARGET_LIST))
AMD_TARGET_ENABLED   = $(filter $(AMD_TARGET),$(SYCL_TARGET_LIST))

CXX                  = $(DPCPP_HOME)/llvm/build/bin/clang++
CUDA_COMPUTE_CAP     = $(firstword $(shell nvidia-smi \
	--query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.'))
CUDA_DETECTED_ARCH   = $(if $(CUDA_COMPUTE_CAP),sm_$(CUDA_COMPUTE_CAP))
CUDA_ARCH            ?= $(CUDA_DETECTED_ARCH)
AMD_AGENT_ENUMERATOR ?= rocm_agent_enumerator
AMD_DETECTED_ARCH    = $(firstword $(filter-out gfx000, \
	$(filter gfx%, $(shell $(AMD_AGENT_ENUMERATOR) 2>/dev/null))))
AMD_GPU_ARCH         ?= $(AMD_DETECTED_ARCH)
AMD_RESOURCE_DIR     = $(shell $(CXX) -print-resource-dir 2>/dev/null)
AMD_LIBSPIRV_DIR     = $(AMD_RESOURCE_DIR)/lib/$(AMD_TARGET)-llvm
AMD_LIBSPIRV_NAME    = libspirv.l64.signed_char.bc
AMD_LIBSPIRV         ?= $(AMD_LIBSPIRV_DIR)/$(AMD_LIBSPIRV_NAME)
AMD_LIBSPIRV_SOURCE_DIR = $(patsubst %/,%,$(abspath \
	$(dir $(AMD_LIBSPIRV))))
AMD_LIBSPIRV_ALIAS_DIR = $(AMD_RESOURCE_DIR)/lib/$(AMD_TARGET)
AMD_LIBSPIRV_ALIAS   = $(AMD_LIBSPIRV_ALIAS_DIR)/$(AMD_LIBSPIRV_NAME)

ifneq ($(NVIDIA_TARGET_ENABLED),)
NVIDIA_TARGET_FLAGS = \
	-Xsycl-target-backend=$(NVIDIA_TARGET) \
	--cuda-gpu-arch=$(CUDA_ARCH)
endif

ifneq ($(AMD_TARGET_ENABLED),)
AMD_TARGET_FLAGS = \
	-Xsycl-target-backend=$(AMD_TARGET) \
	--offload-arch=$(AMD_GPU_ARCH)
AMD_LINK_FLAGS = -Xoffload-linker=$(AMD_TARGET) \
	'--lto-newpm-passes=globaloffset,lto<O3>'
endif

CXXFLAGS = -std=c++17 -O3 -fsycl -fsycl-targets=$(SYCL_TARGETS) \
	$(NVIDIA_TARGET_FLAGS) $(AMD_TARGET_FLAGS)
LDFLAGS = $(AMD_LINK_FLAGS)

ENABLED_DEVICE_SELECTORS :=
ifneq ($(NVIDIA_TARGET_ENABLED),)
ENABLED_DEVICE_SELECTORS += cuda:*
endif
ifneq ($(SPIRV_TARGET_ENABLED),)
ENABLED_DEVICE_SELECTORS += level_zero:* opencl:*
endif
ifneq ($(AMD_TARGET_ENABLED),)
ENABLED_DEVICE_SELECTORS += hip:*
endif

ENABLED_DEVICE_SELECTORS := $(strip $(ENABLED_DEVICE_SELECTORS))
DEFAULT_DEVICE_SELECTOR = $(subst $(space),;,$(ENABLED_DEVICE_SELECTORS))
ONEAPI_DEVICE_SELECTOR ?= $(DEFAULT_DEVICE_SELECTOR)

HOSTILE_HOST_FLAGS = \
	-Xarch_host -fdenormal-fp-math=positive-zero \
	-Xarch_host -fno-signed-zeros \
	-Xarch_host -ffp-eval-method=double

# Google Test (git submodule)
GTEST_DIR    = third_party/googletest
GTEST_BUILD  = $(CURDIR)/build/gtest-build
GTEST_PREFIX = $(CURDIR)/build/gtest
GTEST_LIB    = $(GTEST_PREFIX)/lib
GTEST_INC    = $(GTEST_PREFIX)/include

BUILD_CONFIG        = $(CURDIR)/build/.sycl-build-config
TEST_VARIANT_DIR    = $(CURDIR)/build/test-variants
HOSTILE_HOST_TEST   = $(TEST_VARIANT_DIR)/repro_test_hostile_host
DEVICE_DENORM_TEST  = $(TEST_VARIANT_DIR)/repro_test_device_denorm
DEVICE_NOSZERO_TEST = $(TEST_VARIANT_DIR)/repro_test_device_noszero
DEVICE_EVAL_LOG     = $(TEST_VARIANT_DIR)/device_eval.log

# Runtime library path for execution
export LD_LIBRARY_PATH := $(DPCPP_HOME)/llvm/build/lib:$(LD_LIBRARY_PATH)
export ONEAPI_DEVICE_SELECTOR

# ---- Targets ------------------------------------------------

.PHONY: all example test test-validation test-hostile-host \
	test-unsafe-device test-device-denorm test-device-noszero \
	test-device-eval run clean gtest check-config prepare-toolchain FORCE

all: example repro_test

example: repro_example

repro_example: example.cpp repro_sum.hpp Makefile $(BUILD_CONFIG) | \
		prepare-toolchain
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

repro_test: repro_test.cpp repro_sum.hpp Makefile $(BUILD_CONFIG) | \
		prepare-toolchain gtest
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -isystem $(ONEDPL_INC) \
		-I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

run: repro_example
	./repro_example

test: repro_test
	./repro_test

check-config:
	@echo "SYCL targets: $(SYCL_TARGETS)"
	@echo "Device selector: $(ONEAPI_DEVICE_SELECTOR)"
ifneq ($(NVIDIA_TARGET_ENABLED),)
	@if [ -z "$(CUDA_ARCH)" ]; then \
		echo "Unable to detect an NVIDIA GPU architecture."; \
		echo "Set CUDA_ARCH explicitly, for example sm_86."; \
		exit 2; \
	fi
	@echo "NVIDIA target: $(NVIDIA_TARGET) ($(CUDA_ARCH))"
endif
ifneq ($(AMD_TARGET_ENABLED),)
	@if [ -z "$(AMD_GPU_ARCH)" ]; then \
		echo "Unable to detect an AMD GPU architecture."; \
		echo "Set AMD_GPU_ARCH explicitly, for example gfx1102."; \
		exit 2; \
	fi
	@if [ ! -f "$(AMD_LIBSPIRV)" ]; then \
		echo "AMD libspirv not found: $(AMD_LIBSPIRV)"; \
		echo "Set AMD_LIBSPIRV to the signed-char bitcode file."; \
		exit 2; \
	fi
	@echo "AMD target: $(AMD_TARGET) ($(AMD_GPU_ARCH))"
	@echo "AMD libspirv: $(AMD_LIBSPIRV)"
endif

prepare-toolchain: check-config
ifneq ($(AMD_TARGET_ENABLED),)
	@if [ ! -f "$(AMD_LIBSPIRV_ALIAS)" ]; then \
		if [ -e "$(AMD_LIBSPIRV_ALIAS_DIR)" ] || \
			[ -L "$(AMD_LIBSPIRV_ALIAS_DIR)" ]; then \
			echo "Invalid AMD libspirv path: $(AMD_LIBSPIRV_ALIAS_DIR)"; \
			exit 2; \
		fi; \
		if ! ln -s "$(AMD_LIBSPIRV_SOURCE_DIR)" \
				"$(AMD_LIBSPIRV_ALIAS_DIR)"; then \
			echo "Unable to create the AMD libspirv alias."; \
			exit 2; \
		fi; \
		echo "Created AMD libspirv alias: $(AMD_LIBSPIRV_ALIAS_DIR)"; \
	fi
endif

test-validation: test-hostile-host test-unsafe-device

test-hostile-host: $(HOSTILE_HOST_TEST)
	$< "--gtest_filter=-*Bench*" --gtest_brief=1

test-unsafe-device: test-device-denorm test-device-noszero test-device-eval

test-device-denorm: $(DEVICE_DENORM_TEST)
	$< "--gtest_filter=*UnsafeDeviceEnvironmentRejected*" --gtest_brief=1

test-device-noszero: $(DEVICE_NOSZERO_TEST)
	$< "--gtest_filter=*UnsafeDeviceEnvironmentRejected*" --gtest_brief=1

test-device-eval: | prepare-toolchain $(TEST_VARIANT_DIR)
	@if $(CXX) $(CXXFLAGS) -ffp-eval-method=double -fsyntax-only \
		example.cpp > $(DEVICE_EVAL_LOG) 2>&1; then \
		cat $(DEVICE_EVAL_LOG); \
		exit 1; \
	fi
	@grep -q "device floating-point expressions must evaluate" \
		$(DEVICE_EVAL_LOG)

$(HOSTILE_HOST_TEST): repro_test.cpp repro_sum.hpp Makefile \
		$(BUILD_CONFIG) | gtest prepare-toolchain $(TEST_VARIANT_DIR)
	$(CXX) $(CXXFLAGS) $(HOSTILE_HOST_FLAGS) $(LDFLAGS) \
		-isystem $(ONEDPL_INC) -I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

$(DEVICE_DENORM_TEST): repro_test.cpp repro_sum.hpp Makefile \
		$(BUILD_CONFIG) | gtest prepare-toolchain $(TEST_VARIANT_DIR)
	$(CXX) $(CXXFLAGS) -fdenormal-fp-math=positive-zero \
		-DADN_TEST_EXPECT_DEVICE_REJECTION $(LDFLAGS) \
		-isystem $(ONEDPL_INC) -I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

$(DEVICE_NOSZERO_TEST): repro_test.cpp repro_sum.hpp Makefile \
		$(BUILD_CONFIG) | gtest prepare-toolchain $(TEST_VARIANT_DIR)
	$(CXX) $(CXXFLAGS) -fno-signed-zeros \
		-DADN_TEST_EXPECT_DEVICE_REJECTION $(LDFLAGS) \
		-isystem $(ONEDPL_INC) -I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

$(TEST_VARIANT_DIR):
	mkdir -p $@

# ---- Build configuration ------------------------------------

FORCE:

$(BUILD_CONFIG): FORCE | $(CURDIR)/build
	@config_tmp="$@.tmp"; \
	printf '%s\n' \
		"CXX=$(CXX)" \
		"CXXFLAGS=$(CXXFLAGS)" \
		"LDFLAGS=$(LDFLAGS)" \
		"ONEDPL_INC=$(ONEDPL_INC)" > "$$config_tmp"; \
	if cmp -s "$$config_tmp" "$@"; then \
		rm -f "$$config_tmp"; \
	else \
		mv "$$config_tmp" "$@"; \
	fi

$(CURDIR)/build:
	mkdir -p $@

# ---- Google Test build (one-time) ---------------------------

gtest: $(GTEST_LIB)/libgtest.a

$(GTEST_LIB)/libgtest.a: $(GTEST_DIR)/CMakeLists.txt
	@echo "Building Google Test..."
	@mkdir -p $(GTEST_BUILD)
	@cmake -S $(GTEST_DIR) -B $(GTEST_BUILD) \
		-DCMAKE_CXX_COMPILER=g++ \
		-DCMAKE_INSTALL_PREFIX=$(GTEST_PREFIX) \
		> /dev/null 2>&1
	@$(MAKE) -C $(GTEST_BUILD) -j$$(nproc) > /dev/null 2>&1
	@$(MAKE) -C $(GTEST_BUILD) install > /dev/null 2>&1
	@echo "Google Test installed to $(GTEST_PREFIX)/"

# ---- Clean ---------------------------------------------------

clean:
	rm -f repro_example repro_test
	rm -rf build
