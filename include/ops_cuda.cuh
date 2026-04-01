/**
 * ops_cuda.cuh - CUDA GPU 算子
 *
 * 这些是 ops_cpu.h 中所有算子的 GPU 版本。
 * GPU 之所以比 CPU 快得多，是因为它有成千上万个核心可以并行计算。
 *
 * CUDA 编程模型的核心概念：
 *
 * 1. 线程层次结构：
 *    Grid（网格）→ Block（线程块）→ Thread（线程）
 *    - Thread：最小执行单元，一个线程做一小块计算
 *    - Block：一组线程（最多 1024 个），同一 Block 内的线程可以共享内存、同步
 *    - Grid：所有 Block 的集合
 *
 * 2. 内存层次结构（从快到慢）：
 *    寄存器（Register）→ 共享内存（Shared Memory）→ L1/L2 Cache → 全局内存（Global Memory）
 *    - 寄存器：每个线程私有，最快（1 cycle）
 *    - 共享内存：同一 Block 内共享，很快（~5 cycles），大小有限（通常 48KB/96KB）
 *    - 全局内存（显存）：所有线程可访问，最慢（~400 cycles），但容量最大
 *
 * 3. 性能优化的关键：
 *    - 合并访存（Coalesced Access）：相邻线程访问相邻内存地址
 *    - 共享内存利用：减少全局内存访问次数
 *    - 占用率（Occupancy）：让尽可能多的线程同时运行
 *    - 减少 warp 分歧（divergence）：同一 warp 的线程走相同分支
 *
 * 注意：本文件需要 NVIDIA GPU 和 CUDA 工具链才能编译。
 * 在没有 GPU 的环境中，项目会自动使用 CPU 算子。
 */

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace minillm {
namespace cuda {

// ============================================================================
// 1. GEMV（矩阵-向量乘法）
//
// GPU 上的实现策略：
// - 每个线程块（Block）负责计算输出向量的一个元素
// - Block 内的线程合作完成一行的点积（利用共享内存做 reduction）
// - 使用共享内存缓存输入向量 a，减少全局内存访问
//
// 为什么不用 cuBLAS？
// 对于推理场景（batch_size=1），自己写的简单 kernel 通常够用，
// 而且更灵活（比如可以融合后续操作）。
// 在 batch 较大时，cuBLAS 的 GEMM 会更优。
// ============================================================================
void matmul(float* out, const float* a, const float* B, int N, int K);

// INT8 量化版 GEMV
void matmul_int8(float* out, const float* a, const int8_t* B_int8,
                 const float* scales, int N, int K, int group_size = 128);

// ============================================================================
// 2. RMSNorm（CUDA 版）
//
// GPU 实现策略：
// - 一个 Block 处理整个向量（因为需要计算全局 RMS）
// - Step 1：每个线程计算部分元素的平方和 → 用共享内存做 reduction 求总和
// - Step 2：计算 rms = sqrt(sum / size + eps)
// - Step 3：每个线程负责归一化并乘以 weight
// ============================================================================
void rmsnorm(float* out, const float* x, const float* weight, int size, float eps);

// ============================================================================
// 3. Softmax（CUDA 版）
//
// GPU 实现策略：
// - Step 1：并行找最大值（reduction）
// - Step 2：并行计算 exp(x_i - max)
// - Step 3：并行求 sum（reduction）
// - Step 4：并行除以 sum
// ============================================================================
void softmax(float* x, int size);

// ============================================================================
// 4. RoPE（CUDA 版）
//
// GPU 实现策略：
// - 每个线程处理一对 (x, y) 分量的旋转
// - 所有头可以并行处理（不同 Block）
// ============================================================================
void rope(float* q, float* k, int pos, int head_dim,
          int num_heads, int num_kv_heads, float rope_theta);

// ============================================================================
// 5. SiLU（CUDA 版）
//
// 最简单的 kernel：每个线程处理一个元素
// silu(x) = x / (1 + exp(-x))
// ============================================================================
void silu(float* x, int size);

// ============================================================================
// 6. 逐元素乘法（CUDA 版）
//
// 同样是最简单的 kernel：每个线程一个元素
// ============================================================================
void elementwise_mul(float* out, const float* a, const float* b, int size);

// ============================================================================
// 7. 向量加法（CUDA 版）
// ============================================================================
void add(float* out, const float* a, const float* b, int size);

// ============================================================================
// 8. 量化/反量化（CUDA 版）
// ============================================================================
void quantize_int8(int8_t* out, float* scales, const float* input,
                   int size, int group_size = 128);

void dequantize_int8(float* out, const int8_t* input, const float* scales,
                     int size, int group_size = 128);

// ============================================================================
// 设备内存管理工具函数
// ============================================================================

// 在 GPU 上分配内存
float* device_malloc_float(int size);
int8_t* device_malloc_int8(int size);

// 释放 GPU 内存
void device_free(void* ptr);

// 在 CPU 和 GPU 之间拷贝数据
void copy_to_device(float* dst, const float* src, int size);
void copy_to_host(float* dst, const float* src, int size);

} // namespace cuda
} // namespace minillm

#endif // USE_CUDA
