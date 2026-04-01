/**
 * model.cpp - Transformer 模型前向传播实现
 *
 * 这是整个推理引擎最核心的文件。
 * forward() 函数实现了完整的 Decoder-only Transformer 前向传播。
 *
 * 前向传播的流程（每生成一个 token 执行一次）：
 *
 * 1. Embedding：token_id → 查表得到向量 x [hidden_size]
 * 2. 对每一层（共 N 层）：
 *    a. RMSNorm(x) → xb
 *    b. Q = xb @ W_q + bias_q     （线性投影）
 *    c. K = xb @ W_k + bias_k
 *    d. V = xb @ W_v + bias_v
 *    e. RoPE(Q, K, pos)           （加入位置信息）
 *    f. 将 K, V 存入 KV Cache
 *    g. 对每个注意力头：
 *       - att = Q_head @ K_cache^T / sqrt(head_dim)   （注意力分数）
 *       - att = softmax(att)                           （归一化）
 *       - out_head = att @ V_cache                     （加权求和）
 *    h. out = concat(out_head_0, ..., out_head_n) @ W_o（输出投影）
 *    i. x = x + out                                    （残差连接）
 *    j. RMSNorm(x) → xb
 *    k. gate = silu(xb @ W_gate)                       （SwiGLU 门控）
 *    l. up = xb @ W_up                                 （上投影）
 *    m. ffn_out = (gate * up) @ W_down                 （下投影）
 *    n. x = x + ffn_out                                （残差连接）
 * 3. RMSNorm(x) → x
 * 4. logits = x @ W_lm_head                            （投影到词表大小）
 * 5. 返回 logits
 */

#include "model.h"
#include "sampler.h"
#include "ops_cpu.h"

#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace minillm {

// ============================================================================
// RunState 初始化：分配所有临时缓冲区
// ============================================================================
void RunState::init(const ModelConfig& config) {
    int hidden = config.hidden_size;
    int intermediate = config.intermediate_size;
    int vocab = config.vocab_size;
    int kv_d = config.kv_dim();
    int n_layers = config.num_hidden_layers;

    // 分配激活值缓冲区
    x = Tensor({hidden});
    xb = Tensor({hidden});
    xb2 = Tensor({hidden});

    // 注意力相关缓冲区
    q = Tensor({config.num_attention_heads * config.head_dim()});
    k = Tensor({kv_d});
    v = Tensor({kv_d});
    att = Tensor({config.num_attention_heads, config.max_seq_len});

    // FFN 缓冲区
    hb = Tensor({intermediate});
    hb2 = Tensor({intermediate});

    // 输出 logits
    logits = Tensor({vocab});

    // KV Cache：每一层都需要独立的 KV 缓存
    // 大小 = max_seq_len * kv_dim
    // 对于 Qwen2-0.5B：每层 32768 * 128 * 4 bytes ≈ 16MB
    // 24 层共计 ≈ 384MB
    // 实际使用中 max_seq_len 可以设小一些以节省内存
    key_cache.resize(n_layers);
    value_cache.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        key_cache[i] = Tensor({config.max_seq_len * kv_d});
        value_cache[i] = Tensor({config.max_seq_len * kv_d});
    }

    std::cout << "运行状态已初始化" << std::endl;
    std::cout << "  KV Cache 每层大小: "
              << (config.max_seq_len * kv_d * 4 / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  KV Cache 总大小: "
              << (config.max_seq_len * kv_d * 4 / 1024 / 1024 * n_layers) << " MB" << std::endl;
}

// ============================================================================
// 加载模型权重
//
// 权重名称到模型参数的映射关系（以 Qwen2 为例）：
//   model.embed_tokens.weight → token_embedding_
//   model.layers.{i}.input_layernorm.weight → layers_[i].attn_norm
//   model.layers.{i}.self_attn.q_proj.weight → layers_[i].wq
//   model.layers.{i}.self_attn.q_proj.bias → layers_[i].bq
//   ...
//   model.norm.weight → final_norm_
//   lm_head.weight → lm_head_（如果 tie_word_embeddings=true，则 = token_embedding_）
// ============================================================================
bool Transformer::load_weights(const std::string& model_dir, bool quantize) {
    // 找到所有 safetensors 文件
    auto files = find_safetensors_files(model_dir);
    if (files.empty()) {
        std::cerr << "在 " << model_dir << " 中未找到 safetensors 文件" << std::endl;
        return false;
    }

    // 打开所有 safetensors 文件
    std::vector<SafetensorsFile> sf_files(files.size());
    for (size_t i = 0; i < files.size(); i++) {
        if (!sf_files[i].open(files[i])) {
            return false;
        }
    }

    // 辅助函数：在所有文件中查找指定名称的张量
    // 因为模型可能被拆分到多个 safetensors 文件中
    auto find_and_load = [&](const std::string& name) -> Tensor {
        for (auto& sf : sf_files) {
            if (sf.has_tensor(name)) {
                return sf.load_tensor(name);
            }
        }
        throw std::runtime_error("找不到张量: " + name);
    };

    auto find_and_load_maybe_quantize = [&](const std::string& name) -> Tensor {
        for (auto& sf : sf_files) {
            if (sf.has_tensor(name)) {
                if (quantize) {
                    return sf.load_tensor_quantized(name);
                } else {
                    return sf.load_tensor(name);
                }
            }
        }
        throw std::runtime_error("找不到张量: " + name);
    };

    // 辅助函数：尝试加载，如果不存在返回空张量（用于可选参数如 bias）
    auto try_load = [&](const std::string& name) -> Tensor {
        for (auto& sf : sf_files) {
            if (sf.has_tensor(name)) {
                return sf.load_tensor(name);
            }
        }
        return Tensor();  // 空张量
    };

    try {
        std::cout << "\n开始加载模型权重..." << std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();

        // ---- 加载词嵌入 ----
        token_embedding_ = find_and_load("model.embed_tokens.weight");
        std::cout << "  词嵌入: " << token_embedding_.shape_str() << std::endl;

        // ---- 逐层加载 ----
        int n_layers = config_.num_hidden_layers;
        layers_.resize(n_layers);

        for (int i = 0; i < n_layers; i++) {
            std::string prefix = "model.layers." + std::to_string(i);

            // Attention Norm
            layers_[i].attn_norm = find_and_load(prefix + ".input_layernorm.weight");

            // Attention 投影权重（这些是计算量最大的，可以选择量化）
            layers_[i].wq = find_and_load_maybe_quantize(prefix + ".self_attn.q_proj.weight");
            layers_[i].wk = find_and_load_maybe_quantize(prefix + ".self_attn.k_proj.weight");
            layers_[i].wv = find_and_load_maybe_quantize(prefix + ".self_attn.v_proj.weight");
            layers_[i].wo = find_and_load_maybe_quantize(prefix + ".self_attn.o_proj.weight");

            // Attention 偏置（Qwen2 有，LLaMA 没有）
            if (config_.has_bias) {
                layers_[i].bq = try_load(prefix + ".self_attn.q_proj.bias");
                layers_[i].bk = try_load(prefix + ".self_attn.k_proj.bias");
                layers_[i].bv = try_load(prefix + ".self_attn.v_proj.bias");
            }

            // FFN Norm
            layers_[i].ffn_norm = find_and_load(prefix + ".post_attention_layernorm.weight");

            // FFN 权重
            layers_[i].w_gate = find_and_load_maybe_quantize(prefix + ".mlp.gate_proj.weight");
            layers_[i].w_up = find_and_load_maybe_quantize(prefix + ".mlp.up_proj.weight");
            layers_[i].w_down = find_and_load_maybe_quantize(prefix + ".mlp.down_proj.weight");

            if ((i + 1) % 8 == 0 || i == n_layers - 1) {
                std::cout << "  已加载 " << (i + 1) << "/" << n_layers << " 层" << std::endl;
            }
        }

        // ---- 最终归一化 ----
        final_norm_ = find_and_load("model.norm.weight");

        // ---- LM Head（输出投影）----
        if (config_.tie_word_embeddings) {
            // 共享权重：lm_head 就是 embed_tokens 的转置使用
            // 不需要额外加载，直接复用 token_embedding_ 的数据
            // 用 wrap 创建一个不拥有数据的引用
            lm_head_ = Tensor::wrap(token_embedding_.data, token_embedding_.shape);
            std::cout << "  LM Head: 与词嵌入共享权重" << std::endl;
        } else {
            lm_head_ = find_and_load_maybe_quantize("lm_head.weight");
            std::cout << "  LM Head: " << lm_head_.shape_str() << std::endl;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        std::cout << "模型权重加载完成，耗时 " << elapsed << " 秒" << std::endl;

        if (quantize) {
            std::cout << "已启用 INT8 量化" << std::endl;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "加载权重失败: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 加载完整模型（配置 + 分词器 + 权重）
// ============================================================================
bool Transformer::load(const std::string& model_dir, bool quantize) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Mini LLM Engine - 加载模型" << std::endl;
    std::cout << "  模型目录: " << model_dir << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 加载配置
    try {
        config_ = load_model_config(model_dir);
    } catch (const std::exception& e) {
        std::cerr << "加载配置失败: " << e.what() << std::endl;
        return false;
    }

    // 限制 max_seq_len 以节省内存（KV Cache 占用与此成正比）
    // 对于演示，2048 足够了
    if (config_.max_seq_len > 2048) {
        std::cout << "将 max_seq_len 从 " << config_.max_seq_len
                  << " 限制为 2048（节省内存）" << std::endl;
        config_.max_seq_len = 2048;
    }

    // 2. 加载分词器
    std::string tokenizer_path = model_dir + "/tokenizer.json";
    if (!tokenizer_.load(tokenizer_path)) {
        std::cerr << "加载分词器失败" << std::endl;
        return false;
    }

    // 用 config.json 中的 eos/bos token ID 覆盖自动检测的值
    // config.json 的定义更可靠
    if (config_.eos_token_id >= 0) {
        tokenizer_.set_eos_token_id(config_.eos_token_id);
        std::cout << "  EOS token ID (来自 config.json): " << config_.eos_token_id << std::endl;
    }
    if (config_.bos_token_id >= 0) {
        tokenizer_.set_bos_token_id(config_.bos_token_id);
    }

    // 3. 初始化运行状态（分配临时缓冲区和 KV Cache）
    state_.init(config_);

    // 4. 加载权重
    quantized_ = quantize;
    if (!load_weights(model_dir, quantize)) {
        return false;
    }

    // 5. 自动检测是否是 Instruct 模型
    // 检查目录名是否包含 "instruct"（不区分大小写）
    {
        std::string dir_lower = model_dir;
        std::transform(dir_lower.begin(), dir_lower.end(), dir_lower.begin(), ::tolower);
        if (dir_lower.find("instruct") != std::string::npos) {
            is_instruct_ = true;
        }
    }

    std::cout << "\n模型加载完毕！" << std::endl;
    if (is_instruct_) {
        std::cout << "  模式: Instruct（对话模式，自动应用 chat template）" << std::endl;
    } else {
        std::cout << "  模式: Base（续写模式）" << std::endl;
        std::cout << "  提示: 如需对话效果，请下载 Instruct 版本模型" << std::endl;
    }
    return true;
}

// ============================================================================
// 前向传播 - 整个推理引擎的核心
//
// 输入：token_id（当前 token 的 ID）、pos（在序列中的位置）
// 输出：logits 向量的指针（长度 vocab_size）
//
// 推理时每次只处理一个 token（增量推理），
// 利用 KV Cache 避免重复计算之前位置的 K 和 V。
// ============================================================================
float* Transformer::forward(int token_id, int pos) {
    int hidden = config_.hidden_size;
    int head_dim = config_.head_dim();
    int kv_dim = config_.kv_dim();
    int num_heads = config_.num_attention_heads;
    int num_kv_heads = config_.num_kv_heads;
    int num_kv_groups = config_.num_kv_groups();
    int n_layers = config_.num_hidden_layers;
    float eps = config_.rms_norm_eps;

    // ======== Step 1: Embedding 查表 ========
    // 从词嵌入矩阵中取出 token_id 对应的行
    // 就像查字典一样：token_id 是页码，取出这一页的内容（一个 hidden_size 维向量）
    float* embed = token_embedding_.row_ptr(token_id);
    std::memcpy(state_.x.data, embed, hidden * sizeof(float));

    // ======== Step 2: 逐层 Transformer Block ========
    for (int layer = 0; layer < n_layers; layer++) {
        LayerWeights& w = layers_[layer];

        // ---- 2a: Attention 子层 ----

        // RMSNorm 归一化
        cpu::rmsnorm(state_.xb.data, state_.x.data,
                     w.attn_norm.data, hidden, eps);

        // 计算 Q, K, V 投影
        // Q = xb @ W_q^T + bias_q
        // K = xb @ W_k^T + bias_k
        // V = xb @ W_v^T + bias_v
        if (quantized_ && w.wq.dtype == DType::INT8) {
            // 使用量化版矩阵乘法
            int q_dim = num_heads * head_dim;
            int num_groups_q = (hidden + 127) / 128;
            cpu::matmul_int8(state_.q.data, state_.xb.data,
                            w.wq.data_int8, w.wq.scales, q_dim, hidden);
            cpu::matmul_int8(state_.k.data, state_.xb.data,
                            w.wk.data_int8, w.wk.scales, kv_dim, hidden);
            cpu::matmul_int8(state_.v.data, state_.xb.data,
                            w.wv.data_int8, w.wv.scales, kv_dim, hidden);
        } else {
            cpu::matmul(state_.q.data, state_.xb.data, w.wq.data,
                       num_heads * head_dim, hidden);
            cpu::matmul(state_.k.data, state_.xb.data, w.wk.data,
                       kv_dim, hidden);
            cpu::matmul(state_.v.data, state_.xb.data, w.wv.data,
                       kv_dim, hidden);
        }

        // 加偏置（如果有的话，Qwen2 有 attention bias）
        if (w.bq.data) {
            cpu::add(state_.q.data, state_.q.data, w.bq.data, num_heads * head_dim);
        }
        if (w.bk.data) {
            cpu::add(state_.k.data, state_.k.data, w.bk.data, kv_dim);
        }
        if (w.bv.data) {
            cpu::add(state_.v.data, state_.v.data, w.bv.data, kv_dim);
        }

        // 旋转位置编码（RoPE）
        // 让模型知道当前 token 在位置 pos
        cpu::rope(state_.q.data, state_.k.data, pos, head_dim,
                  num_heads, num_kv_heads, config_.rope_theta);

        // 将当前 K, V 存入 KV Cache
        // key_cache[layer] 的第 pos 个位置存入当前的 K
        float* kc = state_.key_cache[layer].data + pos * kv_dim;
        float* vc = state_.value_cache[layer].data + pos * kv_dim;
        std::memcpy(kc, state_.k.data, kv_dim * sizeof(float));
        std::memcpy(vc, state_.v.data, kv_dim * sizeof(float));

        // ---- 2b: 计算注意力 ----
        // 对每个 Q 头，计算它与所有已缓存的 K 的注意力分数

        // 临时存储注意力输出（重用 xb2 缓冲区）
        std::memset(state_.xb2.data, 0, hidden * sizeof(float));

        for (int h = 0; h < num_heads; h++) {
            // 当前 Q 头的数据
            float* q_head = state_.q.data + h * head_dim; // 首地址 + 某个Q头的偏移

            // GQA: 确定这个 Q 头对应哪个 KV 头
            // 例如 num_heads=14, num_kv_heads=2: 头 0-6 用 KV 头 0，头 7-13 用 KV 头 1
            int kv_head = h / num_kv_groups;

            // 注意力分数指针（第 h 个头，长度为 pos+1）
            float* att = state_.att.data + h * config_.max_seq_len; // dimension: 预分配[max_seq_len]做增量推理，但我们只使用前 pos+1 个位置的分数

            // 计算 Q @ K^T：当前 Q 和所有已缓存的 K 做点积
            for (int t = 0; t <= pos; t++) {
                float* k_t = state_.key_cache[layer].data + t * kv_dim + kv_head * head_dim; // 首地址 + 某个位置 * KV头总维度 + 当前位置某个KV头的偏移

                // 点积
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_head[d] * k_t[d];
                }
                // 除以 sqrt(head_dim) 进行缩放
                // 为什么要缩放？如果不缩放，当 head_dim 很大时，
                // 点积值会很大，softmax 后分布会过于尖锐（一个接近 1，其他接近 0）
                att[t] = score / std::sqrt(static_cast<float>(head_dim));
            }

            // Softmax: 将注意力分数转为概率分布（和为 1）
            cpu::softmax(att, pos + 1);

            // 加权求和 V：att @ V_cache
            // output_head = sum(att[t] * V[t]) for t in [0, pos]
            float* out_head = state_.xb2.data + h * head_dim;
            for (int t = 0; t <= pos; t++) {
                float* v_t = state_.value_cache[layer].data + t * kv_dim + kv_head * head_dim;
                float w = att[t];
                for (int d = 0; d < head_dim; d++) {
                    out_head[d] += w * v_t[d];
                }
            }
        }

        // 输出投影 O：将多头注意力的输出合并并投影
        // xb = concat(head_0, head_1, ...) @ W_o
        if (quantized_ && w.wo.dtype == DType::INT8) {
            cpu::matmul_int8(state_.xb.data, state_.xb2.data,
                            w.wo.data_int8, w.wo.scales, hidden, hidden);
        } else {
            cpu::matmul(state_.xb.data, state_.xb2.data, w.wo.data, hidden, hidden);
        }

        // 残差连接：x = x + attention_output
        cpu::add(state_.x.data, state_.x.data, state_.xb.data, hidden);

        // ---- 2c: FFN 子层 ----

        // RMSNorm
        cpu::rmsnorm(state_.xb.data, state_.x.data,
                     w.ffn_norm.data, hidden, eps);

        // SwiGLU FFN：
        // gate = silu(xb @ W_gate)
        // up   = xb @ W_up
        // out  = (gate * up) @ W_down

        int inter = config_.intermediate_size;

        if (quantized_ && w.w_gate.dtype == DType::INT8) {
            cpu::matmul_int8(state_.hb.data, state_.xb.data,
                            w.w_gate.data_int8, w.w_gate.scales, inter, hidden);
            cpu::matmul_int8(state_.hb2.data, state_.xb.data,
                            w.w_up.data_int8, w.w_up.scales, inter, hidden);
        } else {
            cpu::matmul(state_.hb.data, state_.xb.data, w.w_gate.data, inter, hidden);
            cpu::matmul(state_.hb2.data, state_.xb.data, w.w_up.data, inter, hidden);
        }

        // SiLU 激活函数作用于门控分支
        cpu::silu(state_.hb.data, inter);

        // 门控 * 上投影（逐元素相乘）
        cpu::elementwise_mul(state_.hb.data, state_.hb.data, state_.hb2.data, inter);

        // 下投影
        if (quantized_ && w.w_down.dtype == DType::INT8) {
            cpu::matmul_int8(state_.xb.data, state_.hb.data,
                            w.w_down.data_int8, w.w_down.scales, hidden, inter);
        } else {
            cpu::matmul(state_.xb.data, state_.hb.data, w.w_down.data, hidden, inter);
        }

        // 残差连接：x = x + ffn_output
        cpu::add(state_.x.data, state_.x.data, state_.xb.data, hidden);
    }

    // ======== Step 3: 最终归一化 ========
    cpu::rmsnorm(state_.x.data, state_.x.data,
                 final_norm_.data, hidden, eps);

    // ======== Step 4: LM Head - 投影到词表大小 ========
    // logits = x @ W_lm_head^T
    // logits[i] 表示第 i 个 token 是下一个 token 的"得分"
    if (quantized_ && lm_head_.dtype == DType::INT8) {
        cpu::matmul_int8(state_.logits.data, state_.x.data,
                        lm_head_.data_int8, lm_head_.scales,
                        config_.vocab_size, hidden);
    } else {
        cpu::matmul(state_.logits.data, state_.x.data, lm_head_.data,
                   config_.vocab_size, hidden);
    }

    return state_.logits.data;
}

// ============================================================================
// 应用 Chat Template（对话模板）
//
// Instruct 模型需要特定格式的输入才能正确工作。
// 就像发邮件需要有"收件人"、"主题"一样，对话模型需要明确的角色标记。
//
// Qwen2-Instruct 的格式（ChatML）：
//   <|im_start|>system
//   You are a helpful assistant.<|im_end|>
//   <|im_start|>user
//   你好<|im_end|>
//   <|im_start|>assistant
//   （模型从这里开始生成回复）
//
// 如果不用 chat template，直接输入"你好"，模型不知道该"回答"，
// 只会当作普通文本来续写，效果自然很差。
// ============================================================================
std::string Transformer::apply_chat_template(const std::string& prompt) const {
    // Qwen2 / ChatML 格式
    if (config_.model_type == "qwen2") {
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
               "<|im_start|>user\n" + prompt + "<|im_end|>\n"
               "<|im_start|>assistant\n";
    }

    // LLaMA 3 格式
    if (config_.model_type == "llama") {
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
               "You are a helpful assistant.<|eot_id|>"
               "<|start_header_id|>user<|end_header_id|>\n\n"
               + prompt + "<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    // 未知模型类型，原样返回
    return prompt;
}

// ============================================================================
// 重复惩罚
//
// 对已经生成过的 token，降低其 logit 分数，避免模型陷入重复循环。
//
// 原理：
//   如果某个 token 已经出现过，且其 logit > 0，则除以 penalty
//   如果 logit < 0，则乘以 penalty
//   效果是让已出现的 token 的分数更低（被选中的概率更小）
//
// penalty = 1.0 表示不惩罚，1.1 表示轻微惩罚，1.5 表示强惩罚
// ============================================================================
static void apply_repeat_penalty(float* logits, int vocab_size,
                                  const std::vector<int>& generated_tokens,
                                  float penalty) {
    if (penalty <= 1.0f) return;  // 无惩罚

    for (int token_id : generated_tokens) {
        if (token_id >= 0 && token_id < vocab_size) {
            if (logits[token_id] > 0) {
                logits[token_id] /= penalty;
            } else {
                logits[token_id] *= penalty;
            }
        }
    }
}

// ============================================================================
// 文本生成
//
// 自回归生成的流程：
// 1. 将 prompt 编码为 token ID 序列
// 2. 逐个 token 做前向传播（prefill 阶段：处理整个 prompt）
// 3. 每次用 sampler 选择一个新 token（decode 阶段：生成新文本）
// 4. 直到遇到 EOS token 或达到最大长度
//
// Prefill vs Decode:
//   Prefill：处理用户输入的 prompt，一次处理多个 token（但这里简化为逐个处理）
//   Decode：自回归生成，每次处理上一步生成的 1 个 token
// ============================================================================
std::string Transformer::generate(const std::string& prompt, int max_tokens,
                                   float temperature, float top_p,
                                   float repeat_penalty) {
    // Step 1: 如果是 Instruct 模型，包裹 chat template
    std::string actual_prompt = prompt;
    if (is_instruct_model()) {
        // 文本生成模型需要经过微调才能转化为对话模型，通常需要经过三个步骤：1.准备问答模式的数据进行微调；2.设置奖励模型（Reward Model，RM）对生成的文本进行评分；3.使用强化学习（如 PPO）进一步优化模型，使其生成更符合人类偏好的回答
        actual_prompt = apply_chat_template(prompt); // 这里我们下载的 Instruct 模型已经经过微调，能够理解 chat template 的格式
        std::cout << "[检测到 Instruct 模型，已自动应用 chat template]" << std::endl;
    }

    // Step 2: 编码 prompt
    std::vector<int> prompt_tokens = tokenizer_.encode(actual_prompt);
    if (prompt_tokens.empty()) {
        std::cerr << "提示文本编码后为空" << std::endl;
        return "";
    }

    std::cout << "\n输入 token 数: " << prompt_tokens.size() << std::endl;
    std::cout << "开始生成...\n" << std::endl;

    // Step 3: 创建采样器
    Sampler sampler(temperature, top_p);

    // Step 4: 逐 token 前向传播
    std::string output;
    int total_tokens = static_cast<int>(prompt_tokens.size()) + max_tokens;
    if (total_tokens > config_.max_seq_len) {
        total_tokens = config_.max_seq_len;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int gen_count = 0;

    // 记录已生成的 token（用于重复惩罚）
    std::vector<int> generated_tokens;

    // EOS token ID：用于多个可能的停止标记
    int eos_id = tokenizer_.eos_token_id();
    int im_end_id = tokenizer_.token_id("<|im_end|>");

    int next_token = prompt_tokens[0];

    for (int pos = 0; pos < total_tokens; pos++) {
        // 前向传播
        float* logits = forward(next_token, pos);

        if (pos < static_cast<int>(prompt_tokens.size()) - 1) {
            // Prefill 阶段：还在处理 prompt，下一个 token 已知
            next_token = prompt_tokens[pos + 1];
        } else {
            // 应用重复惩罚
            apply_repeat_penalty(logits, config_.vocab_size,
                                generated_tokens, repeat_penalty);

            // Decode 阶段：从 logits 中采样
            next_token = sampler.sample(logits, config_.vocab_size);
            gen_count++;

            // 检查是否是停止 token
            // 对于 Qwen2-Instruct: <|endoftext|>(151643) 或 <|im_end|>(151645)
            if (next_token == eos_id ||
                (im_end_id >= 0 && next_token == im_end_id)) {
                break;
            }

            // 记录已生成的 token
            generated_tokens.push_back(next_token);

            // 解码并输出
            std::string token_str = tokenizer_.decode_token(next_token);
            output += token_str;

            // 流式输出：每生成一个 token 就打印
            std::cout << token_str << std::flush;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n\n========================================" << std::endl;
    std::cout << "生成完毕！" << std::endl;
    std::cout << "  生成 token 数: " << gen_count << std::endl;
    std::cout << "  总耗时: " << elapsed << " 秒" << std::endl;
    if (gen_count > 0) {
        std::cout << "  速度: " << gen_count / elapsed << " tokens/sec" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return output;
}

} // namespace minillm
