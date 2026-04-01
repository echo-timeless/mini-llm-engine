# ============================================================================
# Makefile - 简单构建脚本（不依赖 CMake）
#
# 用法：
#   make              编译（CPU 版本）
#   make clean        清理编译产物
#   make cuda         编译（CUDA 版本，需要 nvcc）
#
# 注意：推荐使用 CMake 构建，这个 Makefile 是简化版备选方案
# ============================================================================

CXX = g++
NVCC = nvcc

CXXFLAGS = -std=c++17 -O2 -march=native -Wall -Iinclude
NVCCFLAGS = -std=c++17 -O2 -Iinclude -DUSE_CUDA

# 源文件
SRCS = src/main.cpp src/ops_cpu.cpp src/safetensors.cpp \
       src/tokenizer.cpp src/sampler.cpp src/model.cpp

CUDA_SRCS = src/ops_cuda.cu

TARGET = mini-llm-engine

# 默认目标：CPU 版本
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lpthread

# CUDA 版本
cuda: $(SRCS) $(CUDA_SRCS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $^ -lcudart

clean:
	rm -f $(TARGET)

.PHONY: all cuda clean
