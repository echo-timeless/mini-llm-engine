cmake_minimum_required(VERSION 3.14)
project(mini-llm-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 头文件搜索路径
include_directories(${CMAKE_SOURCE_DIR}/include)

# CPU 源文件
set(SOURCES
    src/main.cpp
    src/ops_cpu.cpp
    src/safetensors.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/model.cpp
)

# 创建可执行文件
add_executable(mini-llm-engine ${SOURCES})

# 编译选项
# 重点：Debug 构建不要强制优化，否则断点可能无法命中/行号对不上。
if(MSVC)
    target_compile_options(mini-llm-engine PRIVATE
        /EHsc
        $<$<CONFIG:Debug>:/Od>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:MinSizeRel>:/O1>
    )
else()
    target_compile_options(mini-llm-engine PRIVATE
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:RelWithDebInfo>:-O2>
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:MinSizeRel>:-Os>
        $<$<NOT:$<CONFIG:Debug>>:-march=native>
    )
endif()

# macOS 需要链接 pthread
find_package(Threads REQUIRED)
target_link_libraries(mini-llm-engine Threads::Threads)

# ============================================================================
# CUDA 支持（可选）
#
# 如果系统中有 CUDA Toolkit，自动启用 GPU 加速。
# 没有 CUDA 也能正常编译和运行（使用 CPU 算子）。
#
# 手动启用/禁用：cmake -DUSE_CUDA=ON/OFF ..
# ============================================================================
option(USE_CUDA "启用 CUDA GPU 加速" OFF)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_sources(mini-llm-engine PRIVATE src/ops_cuda.cu)
    target_compile_definitions(mini-llm-engine PRIVATE USE_CUDA)
    target_link_libraries(mini-llm-engine CUDA::cudart)

    # 设置 CUDA 架构（根据你的 GPU 调整）
    # sm_70: V100, sm_75: T4, sm_80: A100, sm_86: RTX 3090, sm_89: RTX 4090
    set_property(TARGET mini-llm-engine PROPERTY CUDA_ARCHITECTURES "70;75;80;86;89")

    message(STATUS "CUDA 已启用")
else()
    message(STATUS "CUDA 未启用，将使用 CPU 算子")
endif()
