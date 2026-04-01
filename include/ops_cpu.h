/**
 * ops_cpu.h - CPU 上的基础算子（操作符）
 *
 * 这些算子是 Transformer 模型的"零件"，每个算子完成一个数学运算。
 * Transformer 的前向传播就是按顺序调用这些算子：
 *
 *   输入 token → Embedding → [RMSNorm → Attention → RMSNorm → FFN] × N 层 → RMSNorm → LM Head → 输出 logits
 *
 * 每个算子都有对应的 CUDA 版本（在 ops_cuda.cuh 中），用于 GPU 加速。
 * CPU 版本作为参考实现和回退方案。
 *
 * 所有算子都是"前向"计算（forward），因为推理只需要前向，不需要反向传播。
 */

#pragma once

#include <cstdint>

namespace minillm {
namespace cpu {

// ============================================================================
// 1. 矩阵乘法（GEMM - General Matrix Multiply）
//
// 这是 Transformer 中最核心、最耗时的操作！
// 模型里几乎所有的"变换"本质上都是矩阵乘法：
//   - Q/K/V 投影：x @ W_q, x @ W_k, x @ W_v
//   - 输出投影：attn_output @ W_o
//   - FFN 层：x @ W_gate, x @ W_up, x @ W_down
//
// out[i] = sum(a[j] * B[i][j]) for j in [0, K)
// 其中 a 是输入向量 [K]，B 是权重矩阵 [N, K]，out 是输出向量 [N]
//
// 为什么用向量×矩阵而不是矩阵×矩阵？
// 因为推理时通常一次处理一个 token（seq_len=1），所以输入是一个向量
// ============================================================================
void matmul(float* out, const float* a, const float* B, int N, int K);

// INT8 量化版矩阵乘法
// 权重用 INT8 存储，计算时需要乘以缩放因子还原
// 优点：内存带宽减半，速度更快（尤其是内存瓶颈的场景）
void matmul_int8(float* out, const float* a, const int8_t* B_int8,
                 const float* scales, int N, int K, int group_size = 128);

// ============================================================================
// 2. RMSNorm（Root Mean Square Layer Normalization）
//
// 对向量做归一化，让每一层的输出保持在合理的数值范围内。
// 如果没有归一化，经过 24 层 Transformer，数值会爆炸（变得极大）或消失（变成 0）。
//
// 公式：
//   rms = sqrt(mean(x^2) + eps)
//   out = (x / rms) * weight
//
// 相比 LayerNorm，RMSNorm 去掉了"减均值"的步骤，计算更快，效果相当。
// 这也是为什么现代 LLM（LLaMA、Qwen）都用 RMSNorm 而不是 LayerNorm。
//
// 参数：
//   x      - 输入向量 [size]
//   weight - 可学习的缩放参数 [size]
//   size   - 向量长度
//   eps    - 防止除零的极小值（通常 1e-6）
// ============================================================================
void rmsnorm(float* out, const float* x, const float* weight, int size, float eps);

// ============================================================================
// 3. Softmax
//
// 把一组任意实数变成概率分布（所有值在 0~1 之间，且和为 1）。
// 用途：
//   - Attention 中计算注意力权重：哪些位置更重要
//   - 最后一步将 logits 转成概率：哪个 token 最可能是下一个
//
// 公式：
//   softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
//
// 为什么要减去 max？
//   数值稳定性！exp(x) 增长极快，x=100 时 exp(x) 就溢出了。
//   减去最大值后，最大的 exp 值为 exp(0)=1，不会溢出。
//   数学上减一个常数不影响 softmax 的结果。
// ============================================================================
void softmax(float* x, int size);

// ============================================================================
// 4. RoPE（Rotary Position Embedding，旋转位置编码）
//
// 让模型知道每个 token 在句子中的位置。
//
// 核心思想：
//   把位置信息"旋转"进向量中。就像时钟的指针，不同时间（位置）对应不同角度。
//   相邻位置的向量之间有固定的旋转关系，所以模型能通过向量的"角度差"
//   感知到 token 之间的相对距离。
//
// 具体做法：
//   将 Q 和 K 向量的相邻两个分量看作一个二维平面上的点 (x, y)，
//   然后按位置 pos 对应的角度进行旋转：
//     new_x = x * cos(θ) - y * sin(θ)
//     new_y = x * sin(θ) + y * cos(θ)
//   其中 θ = pos / (rope_theta^(2i/dim))
//
//   频率随维度指数递减：低维旋转快（捕捉局部关系），高维旋转慢（捕捉远程关系）
//
// 参数：
//   q, k      - Query 和 Key 向量
//   pos       - 当前 token 在序列中的位置
//   head_dim  - 每个注意力头的维度
//   num_heads - Q 的头数
//   num_kv_heads - KV 的头数
//   rope_theta   - 旋转基频（Qwen2 用 1000000）
// ============================================================================
void rope(float* q, float* k, int pos, int head_dim,
          int num_heads, int num_kv_heads, float rope_theta);

// ============================================================================
// 5. SiLU（Sigmoid Linear Unit）激活函数
//
// 公式：silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
//
// SiLU（也叫 Swish）是现代 LLM 中 FFN 层使用的激活函数。
// 相比 ReLU（max(0, x)），SiLU 更平滑，训练效果更好。
//
// 在 Transformer 的 FFN（前馈网络）中，SiLU 是 SwiGLU 结构的一部分：
//   output = silu(x @ W_gate) * (x @ W_up)  // 门控 + 上投影
//   output = output @ W_down                  // 下投影
// ============================================================================
void silu(float* x, int size);

// ============================================================================
// 6. 逐元素乘法
//
// out[i] = a[i] * b[i]
//
// 用于 SwiGLU 中将门控信号和上投影结果逐元素相乘
// ============================================================================
void elementwise_mul(float* out, const float* a, const float* b, int size);

// ============================================================================
// 7. 向量加法（残差连接）
//
// out[i] = a[i] + b[i]
//
// 残差连接（Residual Connection）是 Transformer 的关键设计：
//   output = x + sublayer(x)
//
// 直觉：让信息可以"跳过"某一层直接传到后面，避免深层网络中的梯度消失问题。
// 每一层只需要学习"增量"（residual），而不是完整的变换。
// ============================================================================
void add(float* out, const float* a, const float* b, int size);

// ============================================================================
// 8. INT8 量化与反量化
//
// 量化（Quantize）：将 FP32 权重压缩为 INT8
//   步骤：
//   1. 在每 group_size 个元素中，找到绝对值最大的数 max_abs
//   2. 计算缩放因子 scale = max_abs / 127
//   3. 每个元素：int8_val = round(fp32_val / scale)
//
//   这样 [-max_abs, max_abs] 的范围被映射到 [-127, 127]
//   原始值可以近似还原：fp32_val ≈ int8_val * scale
//
// 为什么要分组（group）？
//   如果整个矩阵用一个 scale，那么如果有个别极大值，
//   其他正常值被压缩到很小的 int8 范围，精度损失严重。
//   分组后每组有自己的 scale，精度更高。group_size 通常为 128。
// ============================================================================
void quantize_int8(int8_t* out, float* scales, const float* input,
                   int size, int group_size = 128);

void dequantize_int8(float* out, const int8_t* input, const float* scales,
                     int size, int group_size = 128);

} // namespace cpu
} // namespace minillm
