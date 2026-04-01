/**
 * ops_cuda.cu - CUDA GPU 算子实现
 *
 * 编译要求：
 *   - NVIDIA GPU（Compute Capability >= 7.0，即 Volta 架构及以上）
 *   - CUDA Toolkit（nvcc 编译器）
 *   - 编译命令示例：nvcc -arch=sm_70 ops_cuda.cu -o ops_cuda.o
 *
 * CUDA 编程的基本套路：
 *   1. 定义 kernel 函数（用 __global__ 修饰，在 GPU 上执行）
 *   2. 在 host（CPU）端配置 Grid 和 Block 的大小
 *   3. 用 <<<grid, block>>> 语法启动 kernel
 *   4. GPU 上的成千上万个线程并行执行 kernel
 *
 * 命名约定：
 *   xxxKernel  - 在 GPU 上执行的 kernel 函数（用 __global__ 修饰）
 *   xxx        - 在 CPU 上调用的封装函数（负责配置参数并启动 kernel）
 */

#ifdef USE_CUDA

#include "ops_cuda.cuh"
#include <cstdio>
#include <cassert>

namespace minillm {
namespace cuda {

// ============================================================================
// GPU 设备内存管理
//
// GPU 有自己独立的内存（显存/VRAM），CPU 不能直接访问。
// 需要用 CUDA API 在 GPU 上分配/释放内存，并在 CPU↔GPU 之间拷贝数据。
// ============================================================================

// 检查 CUDA 调用是否成功的宏
// 每个 CUDA API 调用都应该检查返回值，这在调试时非常重要
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误 at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

float* device_malloc_float(int size) {
    float* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(float)));
    return ptr;
}

int8_t* device_malloc_int8(int size) {
    int8_t* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(int8_t)));
    return ptr;
}

void device_free(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

void copy_to_device(float* dst, const float* src, int size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyHostToDevice));
}

void copy_to_host(float* dst, const float* src, int size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// 1. GEMV Kernel（矩阵-向量乘法）
//
// 并行策略：
//   - 每个 Block 负责计算输出向量的一个元素（即 B 的一行和 a 的点积）
//   - Block 内的 256 个线程合作完成这个点积
//   - 使用共享内存做 parallel reduction（并行归约）
//
// Parallel Reduction（并行归约）是 GPU 编程的经典技术：
//   原理：把求和操作变成树状结构
//   初始：thread0有v0, thread1有v1, ..., thread255有v255
//   第1轮：thread0 += thread128, thread1 += thread129, ...（128个线程工作）
//   第2轮：thread0 += thread64, thread1 += thread65, ...（64个线程工作）
//   ...
//   第8轮：thread0 += thread1（1个线程工作）
//   最终：thread0 持有所有值的总和
//
//   时间复杂度从 O(N) 降到 O(log N)
// ============================================================================

// 每个 Block 的线程数，必须是 2 的幂（reduction 要求）
constexpr int BLOCK_SIZE = 256;

__global__ void matmul_kernel(float* out, const float* a, const float* B,
                               int N, int K) {
    // blockIdx.x 决定了这个 Block 负责输出的哪个元素
    int row = blockIdx.x;
    if (row >= N) return;

    // threadIdx.x 是 Block 内的线程编号（0~255）
    int tid = threadIdx.x;

    // 共享内存：Block 内所有线程共享，用于 reduction
    // __shared__ 告诉编译器这段内存分配在 GPU 的共享内存（SRAM）上
    __shared__ float shared_sum[BLOCK_SIZE];

    // Step 1: 每个线程计算部分点积
    // 线程 tid 负责 a[tid], a[tid+256], a[tid+512], ... 这些位置
    // 这种"跨步访问"模式保证了合并访存（coalesced access）：
    // 相邻线程（tid=0,1,2,...）访问相邻内存地址
    float sum = 0.0f;
    const float* B_row = B + row * K;
    for (int j = tid; j < K; j += BLOCK_SIZE) {
        sum += B_row[j] * a[j];
    }
    shared_sum[tid] = sum;

    // Step 2: Parallel Reduction
    // __syncthreads() 是同步屏障：等待 Block 内所有线程都执行到这里
    __syncthreads();

    // 经典的树状归约：每轮将活跃线程数减半
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: 线程 0 写入最终结果
    if (tid == 0) {
        out[row] = shared_sum[0];
    }
}

void matmul(float* out, const float* a, const float* B, int N, int K) {
    // 启动 N 个 Block（每行一个），每个 Block 有 BLOCK_SIZE 个线程
    // <<<grid_size, block_size>>>
    matmul_kernel<<<N, BLOCK_SIZE>>>(out, a, B, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// INT8 量化 GEMV Kernel
//
// 和 FP32 版本类似，但权重用 INT8 存储。
// 计算时需要将 INT8 值乘以对应的 scale 来"反量化"。
// ============================================================================
__global__ void matmul_int8_kernel(float* out, const float* a,
                                    const int8_t* B_int8, const float* scales,
                                    int N, int K, int group_size) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int num_groups_per_row = (K + group_size - 1) / group_size;

    __shared__ float shared_sum[BLOCK_SIZE];

    float sum = 0.0f;
    const int8_t* B_row = B_int8 + row * K;
    const float* row_scales = scales + row * num_groups_per_row;

    for (int j = tid; j < K; j += BLOCK_SIZE) {
        int group_idx = j / group_size;
        float scale = row_scales[group_idx];
        sum += static_cast<float>(B_row[j]) * scale * a[j];
    }
    shared_sum[tid] = sum;

    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[row] = shared_sum[0];
}

void matmul_int8(float* out, const float* a, const int8_t* B_int8,
                 const float* scales, int N, int K, int group_size) {
    matmul_int8_kernel<<<N, BLOCK_SIZE>>>(out, a, B_int8, scales, N, K, group_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 2. RMSNorm Kernel
//
// 并行策略：
//   用一个 Block 处理整个向量，因为需要计算全局的均方值（需要 reduction）
//   Step 1: 并行计算平方和 → reduction 求总和
//   Step 2: 计算 inv_rms
//   Step 3: 并行归一化并乘以 weight
// ============================================================================
__global__ void rmsnorm_kernel(float* out, const float* x, const float* weight,
                                int size, float eps) {
    int tid = threadIdx.x;

    __shared__ float shared_sum[BLOCK_SIZE];

    // Step 1: 每个线程计算部分平方和
    float sum = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        sum += x[i] * x[i];
    }
    shared_sum[tid] = sum;

    __syncthreads();

    // Reduction 求总和
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }

    // Step 2: 计算 inv_rms（只需要线程 0 来算，然后存到共享内存让所有线程能看到）
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(shared_sum[0] / size + eps);
        // rsqrtf = 1/sqrt(x)，GPU 有专门的硬件指令来快速计算
    }
    __syncthreads();

    // Step 3: 并行归一化
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void rmsnorm(float* out, const float* x, const float* weight, int size, float eps) {
    rmsnorm_kernel<<<1, BLOCK_SIZE>>>(out, x, weight, size, eps);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 3. Softmax Kernel
//
// 三步 reduction：找最大值 → 求 exp 和 → 归一化
// ============================================================================
__global__ void softmax_kernel(float* x, int size) {
    int tid = threadIdx.x;

    __shared__ float shared_data[BLOCK_SIZE];

    // Step 1: 找最大值（max reduction）
    float max_val = -1e30f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, x[i]);
    }
    shared_data[tid] = max_val;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = shared_data[0];
    __syncthreads();

    // Step 2: 计算 exp(x_i - max) 并求和
    float sum = 0.0f;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        x[i] = expf(x[i] - global_max);
        sum += x[i];
    }
    shared_data[tid] = sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_data[tid] += shared_data[tid + stride];
        __syncthreads();
    }
    float global_sum = shared_data[0];
    __syncthreads();

    // Step 3: 归一化
    float inv_sum = 1.0f / global_sum;
    for (int i = tid; i < size; i += BLOCK_SIZE) {
        x[i] *= inv_sum;
    }
}

void softmax(float* x, int size) {
    softmax_kernel<<<1, BLOCK_SIZE>>>(x, size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 4. RoPE Kernel（旋转位置编码）
//
// HuggingFace/Qwen2 使用 "half-split" 式 RoPE：
//   将每个头的维度分成前半和后半：
//     前半：vec[0], vec[1], ..., vec[half_dim-1]
//     后半：vec[half_dim], vec[half_dim+1], ..., vec[head_dim-1]
//   每个频率维度 i（0 到 half_dim-1），配对 (vec[i], vec[i+half_dim]) 做旋转
//
// Grid: num_heads 个 Block（Q 和 K 分别处理）
// Block: head_dim/2 个线程（每个线程处理一对）
// ============================================================================
__global__ void rope_kernel(float* vec, int pos, int head_dim,
                             int num_heads, float rope_theta) {
    int head = blockIdx.x;  // 当前是第几个头
    if (head >= num_heads) return;

    int i = threadIdx.x;  // 频率维度索引（0 到 half_dim-1）
    int half_dim = head_dim / 2;
    if (i >= half_dim) return;

    float* head_data = vec + head * head_dim;

    // 计算旋转角度
    float freq = 1.0f / powf(rope_theta, static_cast<float>(2 * i) / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // half-split 配对：(vec[i], vec[i + half_dim])
    float x = head_data[i];
    float y = head_data[i + half_dim];

    // 旋转并写回
    head_data[i]            = x * cos_val - y * sin_val;
    head_data[i + half_dim] = y * cos_val + x * sin_val;
}

void rope(float* q, float* k, int pos, int head_dim,
          int num_heads, int num_kv_heads, float rope_theta) {
    int threads = head_dim / 2;  // 每对一个线程

    // 对 Q 做旋转（num_heads 个 Block）
    rope_kernel<<<num_heads, threads>>>(q, pos, head_dim, num_heads, rope_theta);
    CUDA_CHECK(cudaGetLastError());

    // 对 K 做旋转（num_kv_heads 个 Block）
    rope_kernel<<<num_kv_heads, threads>>>(k, pos, head_dim, num_kv_heads, rope_theta);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 5. SiLU Kernel
//
// 最简单的 element-wise kernel：每个线程处理一个元素
// 这种 kernel 通常是 memory-bound（内存瓶颈），因为计算量相对内存访问量很小
// ============================================================================
__global__ void silu_kernel(float* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void silu(float* x, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;  // 向上取整
    silu_kernel<<<blocks, threads>>>(x, size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 6. 逐元素乘法 Kernel
// ============================================================================
__global__ void elementwise_mul_kernel(float* out, const float* a,
                                        const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] * b[i];
    }
}

void elementwise_mul(float* out, const float* a, const float* b, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    elementwise_mul_kernel<<<blocks, threads>>>(out, a, b, size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 7. 向量加法 Kernel
// ============================================================================
__global__ void add_kernel(float* out, const float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = a[i] + b[i];
    }
}

void add(float* out, const float* a, const float* b, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(out, a, b, size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 8. INT8 量化 Kernel
// ============================================================================
__global__ void quantize_int8_kernel(int8_t* out, float* scales,
                                      const float* input,
                                      int size, int group_size) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;
    int start = group_id * group_size;

    if (start >= size) return;
    int end = min(start + group_size, size);
    int group_len = end - start;

    // 用共享内存找最大绝对值
    __shared__ float shared_max[BLOCK_SIZE];
    float max_abs = 0.0f;
    for (int i = tid; i < group_len; i += BLOCK_SIZE) {
        max_abs = fmaxf(max_abs, fabsf(input[start + i]));
    }
    shared_max[tid] = max_abs;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    __shared__ float scale;
    if (tid == 0) {
        scale = shared_max[0] / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        scales[group_id] = scale;
    }
    __syncthreads();

    // 量化
    float inv_scale = 1.0f / scale;
    for (int i = tid; i < group_len; i += BLOCK_SIZE) {
        int val = __float2int_rn(input[start + i] * inv_scale);
        val = max(-127, min(127, val));
        out[start + i] = static_cast<int8_t>(val);
    }
}

void quantize_int8(int8_t* out, float* scales, const float* input,
                   int size, int group_size) {
    int num_groups = (size + group_size - 1) / group_size;
    quantize_int8_kernel<<<num_groups, BLOCK_SIZE>>>(out, scales, input, size, group_size);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// INT8 反量化 Kernel
// ============================================================================
__global__ void dequantize_int8_kernel(float* out, const int8_t* input,
                                        const float* scales,
                                        int size, int group_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        int group_id = i / group_size;
        out[i] = static_cast<float>(input[i]) * scales[group_id];
    }
}

void dequantize_int8(float* out, const int8_t* input, const float* scales,
                     int size, int group_size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    dequantize_int8_kernel<<<blocks, threads>>>(out, input, scales, size, group_size);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace minillm

#endif // USE_CUDA
