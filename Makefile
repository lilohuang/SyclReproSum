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
#    DPCPP_HOME      - path to DPC++ installation (default: ~/sycl_workspace)
#    SYCL_TARGETS    - offload targets (default: nvptx64-nvidia-cuda,spir64)
# ============================================================

DPCPP_HOME  ?= $(HOME)/sycl_workspace
SYCL_TARGETS ?= nvptx64-nvidia-cuda,spir64

CXX          = $(DPCPP_HOME)/llvm/build/bin/clang++
CXXFLAGS     = -std=c++17 -O3 -fsycl -fsycl-targets=$(SYCL_TARGETS)

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

TEST_VARIANT_DIR    = $(CURDIR)/build/test-variants
HOSTILE_HOST_TEST   = $(TEST_VARIANT_DIR)/repro_test_hostile_host
DEVICE_DENORM_TEST  = $(TEST_VARIANT_DIR)/repro_test_device_denorm
DEVICE_NOSZERO_TEST = $(TEST_VARIANT_DIR)/repro_test_device_noszero
DEVICE_EVAL_LOG     = $(TEST_VARIANT_DIR)/device_eval.log

# Runtime library path for execution
export LD_LIBRARY_PATH := $(DPCPP_HOME)/llvm/build/lib:$(LD_LIBRARY_PATH)

# ---- Targets ------------------------------------------------

.PHONY: all example test test-validation test-hostile-host \
	test-unsafe-device test-device-denorm test-device-noszero \
	test-device-eval run clean gtest

all: example repro_test

example: repro_example

repro_example: example.cpp repro_sum.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

repro_test: repro_test.cpp repro_sum.hpp | gtest
	$(CXX) $(CXXFLAGS) -I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

run: repro_example
	./repro_example

test: repro_test
	./repro_test

test-validation: test-hostile-host test-unsafe-device

test-hostile-host: $(HOSTILE_HOST_TEST)
	$< "--gtest_filter=-*Bench*" --gtest_brief=1

test-unsafe-device: test-device-denorm test-device-noszero test-device-eval

test-device-denorm: $(DEVICE_DENORM_TEST)
	$< "--gtest_filter=*UnsafeDeviceEnvironmentRejected*" --gtest_brief=1

test-device-noszero: $(DEVICE_NOSZERO_TEST)
	$< "--gtest_filter=*UnsafeDeviceEnvironmentRejected*" --gtest_brief=1

test-device-eval: | $(TEST_VARIANT_DIR)
	@if $(CXX) $(CXXFLAGS) -ffp-eval-method=double -fsyntax-only \
		example.cpp > $(DEVICE_EVAL_LOG) 2>&1; then \
		cat $(DEVICE_EVAL_LOG); \
		exit 1; \
	fi
	@grep -q "device floating-point expressions must evaluate" \
		$(DEVICE_EVAL_LOG)

$(HOSTILE_HOST_TEST): repro_test.cpp repro_sum.hpp Makefile | gtest \
		$(TEST_VARIANT_DIR)
	$(CXX) $(CXXFLAGS) $(HOSTILE_HOST_FLAGS) -I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

$(DEVICE_DENORM_TEST): repro_test.cpp repro_sum.hpp Makefile | gtest \
		$(TEST_VARIANT_DIR)
	$(CXX) $(CXXFLAGS) -fdenormal-fp-math=positive-zero \
		-DADN_TEST_EXPECT_DEVICE_REJECTION -I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

$(DEVICE_NOSZERO_TEST): repro_test.cpp repro_sum.hpp Makefile | gtest \
		$(TEST_VARIANT_DIR)
	$(CXX) $(CXXFLAGS) -fno-signed-zeros \
		-DADN_TEST_EXPECT_DEVICE_REJECTION -I$(GTEST_INC) $< \
		$(GTEST_LIB)/libgtest.a $(GTEST_LIB)/libgtest_main.a \
		-lpthread -o $@

$(TEST_VARIANT_DIR):
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
