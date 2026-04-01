/**
 * tensor.h - 张量（Tensor）类
 *
 * 张量是深度学习中最基本的数据结构，本质上就是一个多维数组。
 * - 0 维张量 = 标量（一个数字）
 * - 1 维张量 = 向量（一行数字）
 * - 2 维张量 = 矩阵（一个表格）
 * - 3 维及以上 = 高维张量
 *
 * 在 LLM 推理中，常见的张量形状：
 * - 权重矩阵：[out_features, in_features]，如 [896, 896]
 * - 词嵌入表：[vocab_size, hidden_size]，如 [151936, 896]
 * - 激活值：  [seq_len, hidden_size]，如 [1, 896]（推理时 seq_len 通常为 1）
 *
 * 这个实现故意保持简单：
 * - 用裸指针管理内存（生产代码会用智能指针或引用计数，但这里为了清晰）
 * - 不支持自动求导（推理不需要）
 * - 支持 CPU 和 GPU 两种存储位置
 */

#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "config.h"

namespace minillm {

class Tensor {
public:
    // ---------- 数据成员 ----------

    float* data = nullptr;         // 数据指针（FP32），指向实际存储数字的内存
    int8_t* data_int8 = nullptr;   // INT8 量化数据指针
    float* scales = nullptr;       // 量化的缩放因子（每个 group 一个 scale）

    std::vector<int> shape;        // 形状，如 {896, 896} 表示 896x896 的矩阵
    int numel = 0;                 // 元素总数 = 所有维度的乘积
    DType dtype = DType::FP32;     // 数据类型
    bool owns_data = false;        // 是否拥有数据的所有权（决定析构时是否释放内存）
    bool on_gpu = false;           // 数据是否在 GPU 上

    // ---------- 构造与析构 ----------

    Tensor() = default;

    // 根据形状创建张量并分配内存
    // 例如：Tensor t({896, 896}) 创建一个 896x896 的矩阵
    Tensor(const std::vector<int>& shape, DType dtype = DType::FP32) {
        this->shape = shape;
        this->dtype = dtype;
        this->numel = 1;
        for (int dim : shape) numel *= dim;

        if (dtype == DType::FP32 || dtype == DType::FP16 || dtype == DType::BF16) {
            // FP16/BF16 在加载时会转换为 FP32 存储
            data = new float[numel]();   // () 表示零初始化
            owns_data = true;
        } else if (dtype == DType::INT8) {
            data_int8 = new int8_t[numel]();
            // 每 group_size 个元素共享一个 scale
            int num_groups = (numel + 127) / 128;  // group_size = 128
            scales = new float[num_groups]();
            owns_data = true;
        }
    }

    // 析构函数：如果拥有数据所有权就释放内存
    ~Tensor() {
        free_data();
    }

    // 禁止拷贝（防止双重释放）
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 允许移动（转移所有权）
    Tensor(Tensor&& other) noexcept {
        move_from(std::move(other));
    }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            free_data();
            move_from(std::move(other));
        }
        return *this;
    }

    // ---------- 数据访问 ----------

    // 获取一维下标处的值
    float& operator[](int index) {
        assert(data != nullptr && index >= 0 && index < numel);
        return data[index];
    }
    float operator[](int index) const {
        assert(data != nullptr && index >= 0 && index < numel);
        return data[index];
    }

    // 获取二维坐标处的值，如 t.at(row, col)
    // 内部将二维坐标转为一维：index = row * cols + col
    float& at(int row, int col) {
        assert(shape.size() == 2);
        return data[row * shape[1] + col];
    }
    float at(int row, int col) const {
        assert(shape.size() == 2);
        return data[row * shape[1] + col];
    }

    // 获取指定行的起始指针（用于矩阵运算中按行访问）
    float* row_ptr(int row) {
        assert(shape.size() >= 2);
        int cols = shape.back();
        return data + row * cols;
    }
    const float* row_ptr(int row) const {
        assert(shape.size() >= 2);
        int cols = shape.back();
        return data + row * cols;
    }

    // ---------- 工具方法 ----------

    // 用外部数据包装成张量（不拥有数据所有权，不会释放）
    // 用于引用 safetensors 文件中 mmap 的数据
    static Tensor wrap(float* data, const std::vector<int>& shape) {
        Tensor t;
        t.data = data;
        t.shape = shape;
        t.numel = 1;
        for (int dim : shape) t.numel *= dim;
        t.dtype = DType::FP32;
        t.owns_data = false;  // 不拥有，析构时不释放
        return t;
    }

    // 将数据全部置零
    void zero() {
        if (data) std::memset(data, 0, numel * sizeof(float));
        if (data_int8) std::memset(data_int8, 0, numel * sizeof(int8_t));
    }

    // 从另一个张量复制数据
    void copy_from(const Tensor& other) {
        assert(numel == other.numel);
        if (data && other.data) {
            std::memcpy(data, other.data, numel * sizeof(float));
        }
    }

    // 返回形状的字符串表示，方便调试
    std::string shape_str() const {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }

private:
    void free_data() {
        if (owns_data) {
            delete[] data;
            delete[] data_int8;
            delete[] scales;
        }
        data = nullptr;
        data_int8 = nullptr;
        scales = nullptr;
    }

    void move_from(Tensor&& other) {
        data = other.data;
        data_int8 = other.data_int8;
        scales = other.scales;
        shape = std::move(other.shape);
        numel = other.numel;
        dtype = other.dtype;
        owns_data = other.owns_data;
        on_gpu = other.on_gpu;
        // 清除源对象的所有权
        other.data = nullptr;
        other.data_int8 = nullptr;
        other.scales = nullptr;
        other.owns_data = false;
    }
};

} // namespace minillm
