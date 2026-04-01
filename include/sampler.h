/**
 * sampler.h - 采样策略
 *
 * 模型的输出是一个 logits 向量（每个 token 一个得分），
 * 采样器决定从中选择哪个 token 作为下一个输出。
 *
 * 常见策略：
 *
 * 1. 贪心采样（Greedy）：
 *    直接选得分最高的 token。确定性的，每次运行结果一样。
 *    缺点：生成的文本重复、无趣。
 *
 * 2. Temperature 采样：
 *    先将 logits 除以 temperature，再做 softmax，然后按概率随机选。
 *    - temperature < 1：分布变尖锐，倾向选高概率 token（更确定）
 *    - temperature = 1：原始分布
 *    - temperature > 1：分布变平坦，更随机（更有创意但可能胡说）
 *
 * 3. Top-P 采样（Nucleus Sampling）：
 *    按概率从高到低排序，累加到超过 p 为止，只在这些 token 中随机选。
 *    例如 top_p=0.9：只在累计概率前 90% 的 token 中选择。
 *    好处：自动调节候选集大小，概率集中时选择少，分散时选择多。
 *
 * 4. Top-K 采样：
 *    只在概率最高的 K 个 token 中选择。
 *    缺点：K 是固定的，不够灵活。实际中 top-p 用得更多。
 */

#pragma once

#include <vector>
#include <random>

namespace minillm {

class Sampler {
public:
    // temperature: 温度参数，控制随机性
    // top_p: nucleus sampling 阈值
    // seed: 随机种子（固定种子可以复现结果）
    Sampler(float temperature = 0.7f, float top_p = 0.9f, unsigned int seed = 42);

    // 从 logits 中采样一个 token ID
    // logits: 模型输出的得分向量 [vocab_size]
    // vocab_size: 词表大小
    int sample(float* logits, int vocab_size);

    // 设置参数
    void set_temperature(float t) { temperature_ = t; }
    void set_top_p(float p) { top_p_ = p; }

private:
    float temperature_;
    float top_p_;
    std::mt19937 rng_;  // Mersenne Twister 随机数生成器

    // 贪心采样：返回最大值的下标
    int sample_greedy(const float* logits, int size);

    // Top-P 采样
    int sample_top_p(float* logits, int size);
};

} // namespace minillm
