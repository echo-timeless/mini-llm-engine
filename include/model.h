/**
 * model.h - Transformer 模型定义
 *
 * 这个文件定义了完整的 Decoder-only Transformer 模型。
 * GPT、LLaMA、Qwen2 等大语言模型都属于这种架构。
 *
 * 模型结构概览：
 * ┌──────────────────────────────────────────────────────────────┐
 * │                    Transformer 模型                          │
 * │                                                              │
 * │  输入 token_id                                               │
 * │      ↓                                                       │
 * │  [Embedding] → 查表得到 token 的向量表示                       │
 * │      ↓                                                       │
 * │  ┌─── Transformer Block × N 层（N=24 for Qwen2-0.5B）────┐  │
 * │  │                                                         │  │
 * │  │  [RMSNorm] → 归一化                                     │  │
 * │  │      ↓                                                  │  │
 * │  │  [Self-Attention] → 让每个 token "看"其他 token          │  │
 * │  │      │  ① x → Q, K, V（三个线性投影）                   │  │
 * │  │      │  ② RoPE 加位置信息                                │  │
 * │  │      │  ③ K, V 存入 KV Cache                            │  │
 * │  │      │  ④ Q @ K^T → 注意力分数                          │  │
 * │  │      │  ⑤ Softmax → 注意力权重                          │  │
 * │  │      │  ⑥ 注意力权重 @ V → 加权求和                     │  │
 * │  │      │  ⑦ 输出投影                                      │  │
 * │  │      ↓                                                  │  │
 * │  │  [残差连接] → x = x + attention_output                  │  │
 * │  │      ↓                                                  │  │
 * │  │  [RMSNorm] → 再次归一化                                  │  │
 * │  │      ↓                                                  │  │
 * │  │  [FFN / MLP] → 非线性变换，增加模型的表达能力             │  │
 * │  │      │  ① gate = silu(x @ W_gate)  // 门控              │  │
 * │  │      │  ② up   = x @ W_up          // 上投影            │  │
 * │  │      │  ③ out  = (gate * up) @ W_down  // 下投影        │  │
 * │  │      ↓                                                  │  │
 * │  │  [残差连接] → x = x + ffn_output                        │  │
 * │  │                                                         │  │
 * │  └─────────────────────────────────────────────────────────┘  │
 * │      ↓                                                       │
 * │  [RMSNorm] → 最终归一化                                       │
 * │      ↓                                                       │
 * │  [LM Head] → 线性投影到词表大小，得到每个 token 的得分          │
 * │      ↓                                                       │
 * │  输出 logits [vocab_size]                                     │
 * └──────────────────────────────────────────────────────────────┘
 *
 * KV Cache 机制：
 *   推理时，模型每次只生成一个 token。
 *   计算 Attention 需要当前 token 的 Q 和所有已生成 token 的 K、V。
 *   如果每次都重新算所有 K、V，就是 O(n^2) 的计算量（n = 已生成长度）。
 *   KV Cache 把之前算过的 K、V 缓存起来，每次只算新 token 的 K、V，
 *   然后和缓存拼接，这样每一步只需 O(n) 的计算。
 *
 * GQA（Grouped Query Attention）：
 *   传统 MHA：Q、K、V 各有 num_heads 个头
 *   GQA：Q 有 num_heads 个头，K/V 只有 num_kv_heads 个头
 *   多个 Q 头共享同一个 KV 头，大幅减少 KV Cache 的显存占用
 *   Qwen2-0.5B：14 个 Q 头，2 个 KV 头 → KV Cache 节省 7 倍
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "config.h"
#include "tensor.h"
#include "safetensors.h"
#include "tokenizer.h"

namespace minillm {

// ============================================================================
// LayerWeights - 一层 Transformer Block 的所有权重
// ============================================================================
struct LayerWeights {
    // --- Attention 部分 ---
    Tensor attn_norm;    // RMSNorm 权重 [hidden_size]

    Tensor wq;           // Q 投影权重 [num_heads * head_dim, hidden_size]
    Tensor wk;           // K 投影权重 [num_kv_heads * head_dim, hidden_size]
    Tensor wv;           // V 投影权重 [num_kv_heads * head_dim, hidden_size]
    Tensor wo;           // 输出投影权重 [hidden_size, num_heads * head_dim]

    Tensor bq;           // Q 偏置 [num_heads * head_dim]（Qwen2 有，LLaMA 没有）
    Tensor bk;           // K 偏置 [num_kv_heads * head_dim]
    Tensor bv;           // V 偏置 [num_kv_heads * head_dim]

    // --- FFN / MLP 部分 ---
    Tensor ffn_norm;     // RMSNorm 权重 [hidden_size]

    Tensor w_gate;       // SwiGLU 门控投影 [intermediate_size, hidden_size]
    Tensor w_up;         // 上投影 [intermediate_size, hidden_size]
    Tensor w_down;       // 下投影 [hidden_size, intermediate_size]
};

// ============================================================================
// RunState - 推理过程中的运行状态（临时缓冲区）
//
// 这些是前向传播过程中需要的临时张量。
// 每次生成一个新 token 时，这些缓冲区会被重复使用。
// ============================================================================
struct RunState {
    Tensor x;            // 当前激活值 [hidden_size]
    Tensor xb;           // 归一化后的激活值（buffer）[hidden_size]
    Tensor xb2;          // 第二个 buffer [hidden_size]

    Tensor q;            // Query 向量 [num_heads * head_dim]
    Tensor k;            // Key 向量 [num_kv_heads * head_dim]
    Tensor v;            // Value 向量 [num_kv_heads * head_dim]

    Tensor att;          // 注意力分数 [num_heads, seq_len]

    Tensor hb;           // FFN 隐藏层 buffer 1 [intermediate_size]
    Tensor hb2;          // FFN 隐藏层 buffer 2 [intermediate_size]

    Tensor logits;       // 输出 logits [vocab_size]

    // KV Cache：存储所有层、所有位置的 K 和 V
    // key_cache[layer][pos * kv_dim ... (pos+1) * kv_dim - 1]
    std::vector<Tensor> key_cache;    // 每层一个 [max_seq_len * kv_dim]
    std::vector<Tensor> value_cache;  // 每层一个 [max_seq_len * kv_dim]

    // 初始化所有缓冲区
    void init(const ModelConfig& config);
};

// ============================================================================
// Transformer - 完整的 Transformer 模型
// ============================================================================
class Transformer {
public:
    Transformer() = default;
    ~Transformer() = default;

    // 从模型目录加载（包括 config.json、safetensors 权重、tokenizer.json）
    bool load(const std::string& model_dir, bool quantize = false);

    // 前向传播：给定 token ID 和位置，计算下一个 token 的 logits
    // 返回 logits 向量（长度 = vocab_size），表示每个 token 的得分
    float* forward(int token_id, int pos);

    // 生成文本
    // prompt: 输入文本
    // max_tokens: 最多生成多少个 token
    // temperature: 控制随机性（0 = 贪心，1 = 正常随机，>1 = 更随机）
    // top_p: nucleus sampling 的阈值
    // repeat_penalty: 重复惩罚系数（1.0=无惩罚，>1.0=惩罚重复）
    std::string generate(const std::string& prompt, int max_tokens = 256,
                         float temperature = 0.7f, float top_p = 0.9f,
                         float repeat_penalty = 1.1f);

    // 判断是否是 Instruct 模型
    // 通过模型目录名是否包含 "instruct"（不区分大小写）来判断
    bool is_instruct_model() const { return is_instruct_; }

    // 外部强制设置是否使用 chat template
    void set_instruct_mode(bool enabled) { is_instruct_ = enabled; }

    // 将用户输入包装成 chat template 格式
    // 例如 Qwen2-Instruct 的格式：
    //   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    //   <|im_start|>user\n{prompt}<|im_end|>\n
    //   <|im_start|>assistant\n
    std::string apply_chat_template(const std::string& prompt) const;

    // 获取配置和分词器（外部可能需要）
    const ModelConfig& config() const { return config_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }

private:
    ModelConfig config_;
    Tokenizer tokenizer_;
    RunState state_;

    // 模型权重
    Tensor token_embedding_;              // 词嵌入 [vocab_size, hidden_size]
    std::vector<LayerWeights> layers_;    // 每层的权重
    Tensor final_norm_;                   // 最终 RMSNorm [hidden_size]
    Tensor lm_head_;                      // 输出投影 [vocab_size, hidden_size]

    bool quantized_ = false;              // 是否使用量化权重
    bool is_instruct_ = false;            // 是否是 Instruct 模型（使用 chat template）

    // 从 safetensors 文件加载权重到模型
    bool load_weights(const std::string& model_dir, bool quantize);
};

} // namespace minillm
