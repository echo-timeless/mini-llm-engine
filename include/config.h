/**
 * config.h - 模型配置定义
 *
 * 这个文件定义了 Transformer 模型的所有超参数。
 * 不同模型（如 Qwen2-0.5B、LLaMA-3.2-1B）只是这些参数的值不同，
 * 架构本身是相同的（都是 Decoder-only Transformer）。
 *
 * 以 Qwen2-0.5B 为例：
 *   - hidden_size = 896          → 每个 token 用 896 维向量表示
 *   - num_attention_heads = 14   → 14 个注意力头
 *   - num_kv_heads = 2           → KV 只用 2 个头（GQA，节省显存）
 *   - num_hidden_layers = 24     → 24 层 Transformer Block 堆叠
 *   - intermediate_size = 4864   → FFN 中间层维度
 *   - vocab_size = 151936        → 词表大小（能识别多少不同的 token）
 */

#pragma once

#include <string>

namespace minillm {

// ============================================================================
// 数据类型枚举
// 模型权重通常用 FP16/BF16 存储以节省空间，推理时转为 FP32 计算
// INT8/INT4 是量化后的类型，用更少的位数表示权重，牺牲少量精度换取速度
// ============================================================================
enum class DType {
    FP32,   // 32 位浮点，精度最高，速度最慢，每个参数占 4 字节
    FP16,   // 16 位浮点，精度够用，每个参数占 2 字节
    BF16,   // Brain Float 16，Google 提出，指数位更多，训练更稳定
    INT8,   // 8 位整数量化，每个参数只占 1 字节，速度快但精度下降
    INT4    // 4 位整数量化，每个参数只占 0.5 字节，极致压缩
};

// ============================================================================
// ModelConfig - 模型超参数配置
//
// 这些参数决定了模型的"形状"，就像建筑的蓝图：
// - hidden_size 决定了模型能记住多少信息（越大越聪明，但越慢）
// - num_hidden_layers 决定了模型的"深度"（层数越多，理解越深）
// - num_attention_heads 决定了模型能同时关注多少种不同的关系
// ============================================================================
struct ModelConfig {
    // ---------- 基本架构参数 ----------

    int vocab_size = 151936;        // 词表大小：模型能识别多少不同的 token
                                     // Qwen2 的词表比 LLaMA（32000）大很多，支持更多语言

    int hidden_size = 896;           // 隐藏层维度：每个 token 的向量表示大小
                                     // 可以理解为模型的"思考宽度"

    int num_hidden_layers = 24;      // Transformer 层数：模型的"思考深度"
                                     // 每一层都会对输入做一次注意力 + FFN 变换

    int num_attention_heads = 14;    // 注意力头数：模型能同时关注多少种关系
                                     // head_dim = hidden_size / num_attention_heads = 64

    int num_kv_heads = 2;            // KV 头数（GQA）：Key 和 Value 用更少的头
                                     // 这是 GQA（Grouped Query Attention）的核心思想：
                                     // Q 有 14 个头，但 K/V 只有 2 个头，
                                     // 每 7 个 Q 头共享 1 个 KV 头
                                     // 好处：大幅减少 KV Cache 的显存占用

    int intermediate_size = 4864;    // FFN 中间层维度
                                     // 通常是 hidden_size 的 ~5.4 倍
                                     // FFN 的结构：hidden → intermediate → hidden

    // ---------- 位置编码参数 ----------

    int max_seq_len = 32768;         // 最大序列长度：模型能处理的最长文本
    float rope_theta = 1000000.0f;   // RoPE 旋转位置编码的基础频率
                                     // 值越大，位置编码变化越慢，支持的序列越长

    // ---------- 归一化参数 ----------

    float rms_norm_eps = 1e-6f;      // RMSNorm 中防止除零的极小值
                                     // LayerNorm 的简化版，去掉了减均值的步骤

    // ---------- 其他 ----------

    bool tie_word_embeddings = true;  // 输入嵌入和输出层是否共享权重
                                      // Qwen2-0.5B 共享（tie=true），即 lm_head = embed_tokens
                                      // 好处：减少参数量

    bool has_bias = true;             // 注意力层是否有偏置项
                                      // Qwen2 有 bias，LLaMA 没有

    std::string model_type = "qwen2"; // 模型类型标识

    int eos_token_id = -1;            // 从 config.json 读取的 EOS token ID
    int bos_token_id = -1;            // 从 config.json 读取的 BOS token ID

    // ---------- 计算得到的参数 ----------

    int head_dim() const { return hidden_size / num_attention_heads; }
        // 每个注意力头的维度 = hidden_size / num_attention_heads
        // 例如 896 / 14 = 64

    int kv_dim() const { return num_kv_heads * head_dim(); }
        // KV 的总维度 = kv 头数 * 每头维度
        // 例如 2 * 64 = 128（远小于 Q 的 896）

    int num_kv_groups() const { return num_attention_heads / num_kv_heads; }
        // 每个 KV 头对应多少个 Q 头
        // 例如 14 / 2 = 7（每个 KV 头被 7 个 Q 头共享）

    // 从 HuggingFace 的 config.json 加载配置
    static ModelConfig from_json(const std::string& json_path);
};

} // namespace minillm
