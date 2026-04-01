# Mini LLM Engine

从零实现的高性能 LLM 推理引擎，使用 C++/CUDA 手写所有核心算子。

## 特性

- **手写核心算子**：GEMV、RMSNorm、Softmax、RoPE、SiLU，提供 CPU 和 CUDA 两套实现
- **Safetensors 加载**：直接解析 HuggingFace 模型格式，支持 FP16/BF16 自动转换
- **BPE 分词器**：从 tokenizer.json 加载，完整实现 BPE 编码/解码，支持特殊 token
- **Chat Template**：自动检测 Instruct 模型并应用对话模板（ChatML 格式）
- **KV Cache**：缓存已计算的 Key/Value，避免重复计算
- **INT8 量化**：按组量化（group_size=128），内存占用减少约 4 倍
- **GQA 支持**：支持 Grouped Query Attention（Qwen2、LLaMA3 等现代模型使用）
- **重复惩罚**：防止模型陷入重复循环
- **零依赖**：不依赖 PyTorch、ONNX 等框架，JSON 解析器也是自己实现的

## 支持模型

- **Qwen2-0.5B-Instruct**（推荐，约 1GB，CPU 可运行，支持对话）
- **Qwen2-1.5B-Instruct**（效果更好，需要 4GB+ 内存）
- **Qwen2.5 系列**（最新版本）
- Qwen2-0.5B（基座模型，仅续写）
- 理论上支持所有 LLaMA 架构变体（LLaMA、Qwen2、Mistral 等）

## 快速开始

### 1. 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

# 如果有 NVIDIA GPU，启用 CUDA
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
```

#### Windows (推荐：MSYS2 UCRT64 + Ninja)

1) 安装工具链（在 MSYS2 UCRT64 环境中执行）：

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja
```

2) 在 PowerShell 或 CMD 里编译：

```bash
D:/msys64/ucrt64/bin/cmake.exe -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=D:/msys64/ucrt64/bin/g++.exe
D:/msys64/ucrt64/bin/cmake.exe --build build -j 8
```

### 2. 下载模型

```bash
# 安装依赖
pip install modelscope

# 下载推荐模型（Qwen2-0.5B-Instruct，使用 ModelScope 国内源）
python3 scripts/download_model.py

# 查看所有可用模型
python3 scripts/download_model.py --list

# 下载更大的模型（效果更好）
python3 scripts/download_model.py --model qwen2-1.5b-instruct

# 使用 HuggingFace 下载
python3 scripts/download_model.py --source huggingface
```

### 3. 运行

```bash
# 对话模式（Instruct 模型自动应用 chat template）
./mini-llm-engine ./models/Qwen2-0.5B-Instruct --prompt "你好"

# Windows:
./build/mini-llm-engine.exe ./models/Qwen2-0.5B-Instruct --prompt "你好"

# 贪心解码（确定性输出）
./mini-llm-engine ./models/Qwen2-0.5B-Instruct --temperature 0.0 --prompt "什么是机器学习"

# INT8 量化（更快，省内存）
./mini-llm-engine ./models/Qwen2-0.5B-Instruct --quantize --prompt "你好"

# 交互模式
./mini-llm-engine ./models/Qwen2-0.5B-Instruct --interactive

# 调节参数
./mini-llm-engine ./models/Qwen2-0.5B-Instruct \
    --prompt "写一首诗" \
    --temperature 0.8 \
    --top-p 0.95 \
    --repeat-penalty 1.2 \
    --max-tokens 200

# Base 模型续写
./mini-llm-engine ./models/Qwen2-0.5B --no-chat --prompt "The capital of France is"
```

### 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--prompt <text>` | 输入提示文本 | "你好，请介绍一下你自己" |
| `--max-tokens <n>` | 最大生成 token 数 | 256 |
| `--temperature <f>` | 温度参数（0=贪心） | 0.7 |
| `--top-p <f>` | Top-P 采样阈值 | 0.9 |
| `--repeat-penalty <f>` | 重复惩罚系数 | 1.1 |
| `--chat` | 强制使用 chat template | 自动检测 |
| `--no-chat` | 强制不使用 chat template | 自动检测 |
| `--quantize` | 启用 INT8 量化 | 关闭 |
| `--interactive` | 交互模式 | 关闭 |

## 项目结构

```
mini-llm-engine/
├── include/
│   ├── json.h          # 极简 JSON 解析器（零依赖）
│   ├── config.h        # 模型配置定义
│   ├── tensor.h        # 张量类
│   ├── safetensors.h   # Safetensors 文件解析
│   ├── tokenizer.h     # BPE 分词器
│   ├── model.h         # Transformer 模型定义
│   ├── sampler.h       # 采样策略
│   ├── ops_cpu.h       # CPU 算子声明
│   └── ops_cuda.cuh    # CUDA 算子声明
├── src/
│   ├── main.cpp        # 主程序
│   ├── model.cpp       # 模型前向传播（核心）
│   ├── ops_cpu.cpp     # CPU 算子实现
│   ├── ops_cuda.cu     # CUDA 算子实现
│   ├── safetensors.cpp # Safetensors 解析实现
│   ├── tokenizer.cpp   # 分词器实现
│   └── sampler.cpp     # 采样器实现
├── scripts/
│   └── download_model.py  # 模型下载脚本（支持多模型 + 国内镜像）
├── CMakeLists.txt
├── Makefile
└── README.md
```

## 架构说明

### 推理流程

```
输入文本 "你好"
    ↓
[Chat Template] 包裹对话格式（Instruct 模型）
    ↓
[Tokenizer] encode → token IDs（特殊 token 整体匹配 + BPE）
    ↓
[Embedding] 查表 → 向量 [hidden_size]
    ↓
[Transformer Block] × 24 层
  ├─ RMSNorm → Attention(Q,K,V + RoPE + KV Cache + GQA) → 残差连接
  └─ RMSNorm → FFN(SwiGLU) → 残差连接
    ↓
[RMSNorm] → [LM Head] → logits [vocab_size]
    ↓
[Repeat Penalty] → [Sampler] Top-P 采样 → next_token_id
    ↓
[Tokenizer] decode → "我是一个AI助手"
    ↓
重复直到 EOS 或 <|im_end|> 或达到最大长度
```

### Base 模型 vs Instruct 模型

| | Base 模型 | Instruct 模型 |
|---|---|---|
| 训练方式 | 预测下一个 token | Base + SFT + RLHF |
| 输入格式 | 原始文本 | Chat Template 包裹 |
| 能力 | 续写文本 | 理解并回答指令 |
| 适用场景 | 文本补全 | 对话、问答 |
| 推荐 | 学习用 | 实际使用 |

### CUDA 算子优化要点

- **GEMV**：每个 Block 计算一行点积，Block 内用 shared memory parallel reduction
- **RMSNorm**：单 Block 处理，两次 reduction（求平方和 + 归一化）
- **Softmax**：三次 reduction（max + exp_sum + normalize）
- **RoPE**：每线程处理一对分量的旋转（half-split 配对方式）
- **Element-wise ops**：标准的一线程一元素模式

### INT8 量化方案

- 按组量化（group_size=128）
- 每组独立的 scale = max_abs / 127
- 支持量化 matmul：计算时在线反量化
- 内存占用约为 FP32 的 1/4
