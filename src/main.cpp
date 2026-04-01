/**
 * main.cpp - Mini LLM Engine 主程序入口
 *
 * 用法：
 *   ./mini-llm-engine <模型目录路径> [选项]
 *
 * 示例：
 *   ./mini-llm-engine ./models/Qwen2-0.5B
 *   ./mini-llm-engine ./models/Qwen2-0.5B --quantize --prompt "你好"
 *   ./mini-llm-engine ./models/Qwen2-0.5B --temperature 0.0 --prompt "1+1="
 *
 * 选项：
 *   --prompt <text>        输入提示文本（默认："你好，请介绍一下你自己"）
 *   --max-tokens <n>       最大生成 token 数（默认：256）
 *   --temperature <float>  温度参数（默认：0.7，0 = 贪心）
 *   --top-p <float>        Top-P 采样阈值（默认：0.9）
 *   --quantize             启用 INT8 量化（节省内存，稍有精度损失）
 *   --interactive          交互模式（循环读取用户输入）
 */

#include "model.h"
#include <iostream>
#include <string>
#include <cstring>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

using namespace minillm;

// 解析命令行参数
struct Args {
    std::string model_dir;
    std::string prompt = "你好，请介绍一下你自己";
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float repeat_penalty = 1.1f;
    bool quantize = false;
    bool interactive = false;
    int force_chat = 0;  // 0=自动, 1=强制开启, -1=强制关闭
};

Args parse_args(int argc, char* argv[]) {
    Args args;

    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <模型目录> [选项]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "选项:" << std::endl;
        std::cerr << "  --prompt <text>        输入提示文本" << std::endl;
        std::cerr << "  --max-tokens <n>       最大生成 token 数" << std::endl;
        std::cerr << "  --temperature <float>  温度参数（0=贪心）" << std::endl;
        std::cerr << "  --top-p <float>        Top-P 采样阈值" << std::endl;
        std::cerr << "  --repeat-penalty <f>   重复惩罚系数（默认 1.1）" << std::endl;
        std::cerr << "  --chat                 强制使用 chat template" << std::endl;
        std::cerr << "  --no-chat              强制不使用 chat template" << std::endl;
        std::cerr << "  --quantize             启用 INT8 量化" << std::endl;
        std::cerr << "  --interactive          交互模式" << std::endl;
        exit(1);
    }

    args.model_dir = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            args.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--repeat-penalty" && i + 1 < argc) {
            args.repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "--chat") {
            args.force_chat = 1;
        } else if (arg == "--no-chat") {
            args.force_chat = -1;
        } else if (arg == "--quantize") {
            args.quantize = true;
        } else if (arg == "--interactive") {
            args.interactive = true;
        }
    }

    return args;
}

Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    if (argv.size() < 2) {
        std::cerr << "用法: " << (argv.empty() ? "mini-llm-engine" : argv[0])
                  << " <模型目录> [选项]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "选项:" << std::endl;
        std::cerr << "  --prompt <text>        输入提示文本" << std::endl;
        std::cerr << "  --max-tokens <n>       最大生成 token 数" << std::endl;
        std::cerr << "  --temperature <float>  温度参数（0=贪心）" << std::endl;
        std::cerr << "  --top-p <float>        Top-P 采样阈值" << std::endl;
        std::cerr << "  --repeat-penalty <f>   重复惩罚系数（默认 1.1）" << std::endl;
        std::cerr << "  --chat                 强制使用 chat template" << std::endl;
        std::cerr << "  --no-chat              强制不使用 chat template" << std::endl;
        std::cerr << "  --quantize             启用 INT8 量化" << std::endl;
        std::cerr << "  --interactive          交互模式" << std::endl;
        exit(1);
    }

    args.model_dir = argv[1];

    for (size_t i = 2; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "--prompt" && i + 1 < argv.size()) {
            args.prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argv.size()) {
            args.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argv.size()) {
            args.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argv.size()) {
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--repeat-penalty" && i + 1 < argv.size()) {
            args.repeat_penalty = std::stof(argv[++i]);
        } else if (arg == "--chat") {
            args.force_chat = 1;
        } else if (arg == "--no-chat") {
            args.force_chat = -1;
        } else if (arg == "--quantize") {
            args.quantize = true;
        } else if (arg == "--interactive") {
            args.interactive = true;
        }
    }

    return args;
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t* wstr) {
    if (!wstr) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string out;
    out.resize(static_cast<size_t>(len - 1));
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), len, nullptr, nullptr);
    return out;
}

static std::vector<std::string> get_utf8_argv() {
    int argc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv || argc <= 0) {
        return {};
    }

    std::vector<std::string> argv;
    argv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++) {
        argv.push_back(wide_to_utf8(wargv[i]));
    }

    LocalFree(wargv);
    return argv;
}
#endif

// ============================================================================
// 交互模式：循环读取用户输入并生成回复
// ============================================================================
void interactive_mode(Transformer& model, const Args& args) {
    std::cout << "\n====== 交互模式 ======" << std::endl;
    std::cout << "输入文本后按回车生成回复，输入 'quit' 退出" << std::endl;
    std::cout << std::endl;

    while (true) {
        std::cout << "用户> ";
        std::string input;
        std::getline(std::cin, input);

        if (input.empty()) continue;
        if (input == "quit" || input == "exit" || input == "q") break;

        std::cout << "\n助手> ";
        model.generate(input, args.max_tokens, args.temperature, args.top_p,
                       args.repeat_penalty);
        std::cout << std::endl;
    }

    std::cout << "再见！" << std::endl;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char* argv[]) {
#ifdef _WIN32
    // 让 Windows 控制台正确显示/输入 UTF-8（中文日志/交互更友好）
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    // 解析命令行参数
#ifdef _WIN32
    Args args = parse_args(get_utf8_argv());
#else
    Args args = parse_args(argc, argv);
#endif

    // 创建并加载模型
    Transformer model;
    if (!model.load(args.model_dir, args.quantize)) {
        std::cerr << "模型加载失败！" << std::endl;
        return 1;
    }

    // 应用 chat 模式覆盖
    if (args.force_chat == 1) {
        model.set_instruct_mode(true);
        std::cout << "已强制启用 chat template" << std::endl;
    } else if (args.force_chat == -1) {
        model.set_instruct_mode(false);
        std::cout << "已强制关闭 chat template" << std::endl;
    }

    if (args.interactive) {
        // 交互模式
        interactive_mode(model, args);
    } else {
        // 单次生成模式
        std::cout << "\n提示: " << args.prompt << std::endl;
        std::cout << "回复: ";
        model.generate(args.prompt, args.max_tokens,
                       args.temperature, args.top_p, args.repeat_penalty);
    }

    return 0;
}
