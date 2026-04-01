/**
 * ops_cpu.cpp - CPU 算子实现
 *
 * 这是所有数学运算的 CPU 参考实现。
 * 每个函数都尽量写得清晰易懂，牺牲了一些性能优化（如 SIMD）来换取可读性。
 * 生产级别的实现会用 AVX/NEON 等 SIMD 指令来加速。
 */

#include "ops_cpu.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace minillm {
namespace cpu {

// ============================================================================
// 矩阵-向量乘法（GEMV）
//
// 计算 out = B @ a，其中：
//   a: 输入向量 [K]
//   B: 权重矩阵 [N, K]（N 行 K 列）
//   out: 输出向量 [N]
//
// 本质就是 N 次点积运算：out[i] = dot(B[i], a) = sum(B[i][j] * a[j])
//
// 性能分析：
//   计算量 = 2NK（N 个点积，每个 K 次乘加）
//   访存量 = NK（读取整个 B 矩阵）+ K（读取向量 a）+ N（写入 out）
//   对于 Qwen2-0.5B 的 Q 投影：N=896, K=896, 计算量 ≈ 1.6M FLOPs
//
// 这是推理中最耗时的操作。优化方向：
// 1. CPU: 用 SIMD（AVX2 可以一次算 8 个 float）
// 2. GPU: 用 CUDA 并行（每个线程块算一行）
// 3. 量化: 用 INT8 减少内存带宽需求
// ============================================================================
void matmul(float* out, const float* a, const float* B, int N, int K) {
    // 遍历 B 的每一行，计算点积
    for (int i = 0; i < N; i++) {
        // 注意：用 double 累加以减少浮点精度误差
        // float 只有 23 位尾数（约 7 位有效数字），累加 896 个乘积时误差会积累
        // double 有 52 位尾数（约 15 位有效数字），精度足够
        // 经过 24 层 Transformer 后，float 累加的误差可以大到翻转 logit 排名
        double sum = 0.0;
        const float* row = B + i * K;  // B 的第 i 行起始地址

        // 点积：sum = B[i][0]*a[0] + B[i][1]*a[1] + ... + B[i][K-1]*a[K-1]
        for (int j = 0; j < K; j++) {
            sum += static_cast<double>(row[j]) * static_cast<double>(a[j]);
        }
        out[i] = static_cast<float>(sum);
    }
}

// ============================================================================
// INT8 量化矩阵-向量乘法
//
// 和普通 matmul 的区别：权重 B 用 INT8 存储，每 group_size 个元素共享一个 scale。
// 计算时需要"反量化"：float_val = int8_val * scale
//
// 优势：
// - 内存带宽减半（int8 = 1字节，float32 = 4字节）
// - 在内存带宽瓶颈的场景下（推理通常是），速度接近 4 倍提升
//
// 精度损失：
// - 通常在 1% 以内，对生成质量影响很小
// ============================================================================
void matmul_int8(float* out, const float* a, const int8_t* B_int8,
                 const float* scales, int N, int K, int group_size) {

    int num_groups_per_row = (K + group_size - 1) / group_size; // 取上界

    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        const int8_t* row = B_int8 + i * K;
        const float* row_scales = scales + i * num_groups_per_row;

        // 按组计算，每组有自己的 scale
        for (int g = 0; g < num_groups_per_row; g++) {
            int start = g * group_size;
            int end = std::min(start + group_size, K);
            double scale = static_cast<double>(row_scales[g]);

            for (int j = start; j < end; j++) {
                sum += static_cast<double>(row[j]) * scale * static_cast<double>(a[j]);
            }
        }
        out[i] = static_cast<float>(sum);
    }
}

// ============================================================================
// RMSNorm
//
// 步骤：
// 1. 计算输入向量所有元素的平方的均值：mean_sq = sum(x_i^2) / size
// 2. 计算 RMS：rms = sqrt(mean_sq + eps)
// 3. 归一化并乘以权重：out_i = (x_i / rms) * weight_i
//
// eps 的作用：防止当 x 全为 0 时除以 0
// ============================================================================
void rmsnorm(float* out, const float* x, const float* weight, int size, float eps) {
    // Step 1: 计算 x 的均方值（用 double 累加提高精度）
    double sum_sq = 0.0;
    for (int i = 0; i < size; i++) {
        sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }

    // Step 2: 计算归一化系数 1/rms
    float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / size) + eps);

    // Step 3: 归一化并乘以 weight
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// ============================================================================
// Softmax
//
// 数值稳定版本的实现，三步走：
// 1. 找最大值 max（防止 exp 溢出）
// 2. 每个元素减 max 后取 exp
// 3. 除以所有 exp 的和，使总和为 1（概率分布）
// ============================================================================
void softmax(float* x, int size) {
    // Step 1: 找最大值
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Step 2: exp(x_i - max) 并求和
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Step 3: 归一化
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// ============================================================================
// RoPE（旋转位置编码）
//
// 对 Q 和 K 的每一对相邻分量 (x, y) 做二维旋转：
//   new_x = x * cos(θ) - y * sin(θ)
//   new_y = x * sin(θ) + y * cos(θ)
//
// 其中旋转角度 θ = pos * freq，freq = 1 / (rope_theta^(2i/dim))
//
// i 是维度索引（从 0 到 head_dim/2 - 1）
// 低维度的 freq 大 → 旋转快 → 对相近位置敏感（捕捉局部关系）
// 高维度的 freq 小 → 旋转慢 → 对远距离位置敏感（捕捉远程关系）
// ============================================================================
void rope(float* q, float* k, int pos, int head_dim,
          int num_heads, int num_kv_heads, float rope_theta) {

    // HuggingFace/Qwen2 使用的是 "half-split" 式 RoPE：
    //
    // 将每个头的维度分成前半和后半：
    //   前半：q[0], q[1], ..., q[31]
    //   后半：q[32], q[33], ..., q[63]
    //
    // 每个频率维度 i（0 到 31），配对 (q[i], q[i+32]) 做旋转：
    //   new_q[i]      = q[i] * cos(θ_i) - q[i+32] * sin(θ_i)
    //   new_q[i+32]   = q[i+32] * cos(θ_i) + q[i] * sin(θ_i)
    //
    // 其中 θ_i = pos * freq_i，freq_i = 1 / (theta^(2i/dim))
    //
    // 直觉理解：
    //   前半部分存"实部"，后半部分存"虚部"，每对做复数乘法实现旋转
    //   这和传统的相邻配对 (q[0],q[1]),(q[2],q[3])... 在数学上是不同的

    int half_dim = head_dim / 2;

    // 对 Q 的每个头做旋转
    for (int h = 0; h < num_heads; h++) {
        float* q_head = q + h * head_dim;

        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
            float angle = pos * freq;

            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            // 配对 (q[i], q[i + half_dim]) 做旋转
            float x = q_head[i];
            float y = q_head[i + half_dim];

            q_head[i]            = x * cos_val - y * sin_val;
            q_head[i + half_dim] = y * cos_val + x * sin_val;
        }
    }

    // 对 K 的每个头做同样的旋转
    for (int h = 0; h < num_kv_heads; h++) {
        float* k_head = k + h * head_dim;

        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
            float angle = pos * freq;

            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float x = k_head[i];
            float y = k_head[i + half_dim];

            k_head[i]            = x * cos_val - y * sin_val;
            k_head[i + half_dim] = y * cos_val + x * sin_val;
        }
    }
}

// ============================================================================
// SiLU 激活函数
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// 当 x 很大时：sigmoid ≈ 1，silu ≈ x（保持不变）
// 当 x 很小（负数）时：sigmoid ≈ 0，silu ≈ 0（抑制）
// 特点：和 ReLU 类似但更平滑，在 x=0 处可导
// ============================================================================
void silu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// ============================================================================
// 逐元素乘法
// ============================================================================
void elementwise_mul(float* out, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] * b[i];
    }
}

// ============================================================================
// 向量加法（用于残差连接）
// ============================================================================
void add(float* out, const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = a[i] + b[i];
    }
}

// ============================================================================
// INT8 量化
//
// 将 FP32 数据按组量化为 INT8：
// 1. 在每组中找最大绝对值 max_abs
// 2. scale = max_abs / 127（将范围映射到 [-127, 127]）
// 3. int8_val = round(fp32_val / scale)
//
// 为什么是 127 而不是 128？
// 因为 INT8 的范围是 [-128, 127]，为了对称我们用 [-127, 127]
// ============================================================================
void quantize_int8(int8_t* out, float* scales, const float* input,
                   int size, int group_size) {
    int num_groups = (size + group_size - 1) / group_size;

    for (int g = 0; g < num_groups; g++) {
        int start = g * group_size;
        int end = std::min(start + group_size, size);

        // Step 1: 找这一组的最大绝对值
        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = std::fabs(input[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        // Step 2: 计算缩放因子
        // 加一个极小值防止 max_abs = 0 导致除零
        float scale = max_abs / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        scales[g] = scale;

        // Step 3: 量化
        float inv_scale = 1.0f / scale;
        for (int i = start; i < end; i++) {
            int val = static_cast<int>(std::round(input[i] * inv_scale));
            // 裁剪到 [-127, 127] 范围
            val = std::max(-127, std::min(127, val));
            out[i] = static_cast<int8_t>(val);
        }
    }
}

// ============================================================================
// INT8 反量化
//
// 将 INT8 数据还原为 FP32：fp32_val = int8_val * scale
// ============================================================================
void dequantize_int8(float* out, const int8_t* input, const float* scales,
                     int size, int group_size) {
    int num_groups = (size + group_size - 1) / group_size;

    for (int g = 0; g < num_groups; g++) {
        int start = g * group_size;
        int end = std::min(start + group_size, size);
        float scale = scales[g];

        for (int i = start; i < end; i++) {
            out[i] = static_cast<float>(input[i]) * scale;
        }
    }
}

} // namespace cpu
} // namespace minillm
