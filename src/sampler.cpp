/**
 * sampler.cpp - 采样器实现
 *
 * 从模型输出的 logits 中选择下一个 token。
 * logits 是"原始得分"，越高表示模型越觉得这个 token 应该是下一个。
 */

#include "sampler.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace minillm {

Sampler::Sampler(float temperature, float top_p, unsigned int seed)
    : temperature_(temperature), top_p_(top_p), rng_(seed) {}

// ============================================================================
// 采样入口
// ============================================================================
int Sampler::sample(float* logits, int vocab_size) {
    // temperature ≈ 0 时用贪心（选最大）
    if (temperature_ < 1e-6f) {
        return sample_greedy(logits, vocab_size);
    }

    // Step 1: 除以 temperature
    // 直觉：temperature 越低，大概率 token 的优势越明显（分布越尖）
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature_;
    }

    // Step 2: 用 top-p 采样
    return sample_top_p(logits, vocab_size);
}

// ============================================================================
// 贪心采样：直接选最大值
//
// 优点：确定性，适合代码生成等需要精确输出的场景
// 缺点：容易陷入重复模式（因为总是选最可能的 token）
// ============================================================================
int Sampler::sample_greedy(const float* logits, int size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// ============================================================================
// Top-P 采样（Nucleus Sampling）
//
// 算法步骤：
// 1. 对 logits 做 softmax，得到概率分布
// 2. 按概率从大到小排序
// 3. 从概率最大的开始累加，直到累积概率超过 top_p
// 4. 在这些"核心"token 中，按概率随机选择一个
//
// 为什么 Top-P 比 Top-K 好？
// - Top-K（如 K=50）：无论概率分布如何，总是在前 50 个中选
//   → 如果概率集中在前 3 个，那 47 个低概率 token 是噪声
//   → 如果概率很均匀，50 个可能不够
// - Top-P（如 P=0.9）：自适应调节候选集大小
//   → 概率集中时候选集小，分散时候选集大
// ============================================================================
int Sampler::sample_top_p(float* logits, int size) {
    // Step 1: Softmax
    // 先找最大值（数值稳定性）
    float max_val = *std::max_element(logits, logits + size);
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        logits[i] *= inv_sum;
    }

    // Step 2: 创建 (概率, 索引) 对并按概率降序排序
    // 注意：对整个词表排序开销很大。更高效的实现会用 partial_sort。
    // 这里为了代码清晰使用完整排序。
    std::vector<std::pair<float, int>> prob_idx(size);
    for (int i = 0; i < size; i++) {
        prob_idx[i] = {logits[i], i};
    }

    // 部分排序：只排前面概率高的，直到累积概率超过 top_p
    // 用 partial_sort 而不是 sort，因为大部分 token 概率极低，不需要排
    std::partial_sort(
        prob_idx.begin(),
        prob_idx.begin() + std::min(size, 1000),  // 只排前 1000 个
        prob_idx.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Step 3: 累积概率，找到截断点
    float cumsum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < size && i < 1000; i++) {
        cumsum += prob_idx[i].first;
        cutoff = i + 1;
        if (cumsum >= top_p_) break;
    }

    // Step 4: 在截断范围内按概率随机选择
    // 生成 [0, cumsum) 范围内的随机数
    std::uniform_real_distribution<float> dist(0.0f, cumsum);
    float r = dist(rng_);

    float running_sum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        running_sum += prob_idx[i].first;
        if (running_sum >= r) {
            return prob_idx[i].second;
        }
    }

    // 理论上不会执行到这里，但作为安全回退，返回概率最高的 token
    return prob_idx[0].second;
}

} // namespace minillm
