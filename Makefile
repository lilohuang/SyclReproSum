# ============================================================
#  Ahrens-Demmel-Nguyen K-Fold Reproducible Summation - Makefile
# ============================================================
#
#  Targets:
#    make            - build example (repro_example)
#    make test       - build & run Google Test suite
#    make all        - build both example and test
#    make run        - build & run example
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

# Google Test (git submodule)
GTEST_DIR    = third_party/googletest
GTEST_BUILD  = $(CURDIR)/build/gtest-build
GTEST_PREFIX = $(CURDIR)/build/gtest
GTEST_LIB    = $(GTEST_PREFIX)/lib
GTEST_INC    = $(GTEST_PREFIX)/include

# Runtime library path for execution
export LD_LIBRARY_PATH := $(DPCPP_HOME)/llvm/build/lib:$(LD_LIBRARY_PATH)

# ---- Targets ------------------------------------------------

.PHONY: all example test run clean gtest

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
