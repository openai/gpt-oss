cmake_minimum_required(VERSION 3.24)
project(GPTOSS
    VERSION 1.0
    DESCRIPTION "Local GPT-OSS inference"
    LANGUAGES C CXX OBJC)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJC_STANDARD 11)
set(CMAKE_OBJC_STANDARD_REQUIRED ON)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK      Metal      REQUIRED)
find_library(IOKIT_FRAMEWORK      IOKit      REQUIRED)

set(METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/source/accumulate.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/convert.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/embeddings.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/expert_routing_metadata.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/gather_and_accumulate.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/matmul.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/moematmul.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/random.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/rmsnorm.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/rope.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/sample.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/scatter.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/sdpa.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/source/topk.metal
)
set(METAL_LIB default.metallib)

include_directories(BEFORE include source/include)

add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/source/"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/accumulate.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/accumulate.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/convert.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/convert.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/embeddings.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/embeddings.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/expert_routing_metadata.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/expert_routing_metadata.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/matmul.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/matmul.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/moematmul.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/moematmul.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/gather_and_accumulate.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/gather_and_accumulate.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/random.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/random.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/rmsnorm.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/rmsnorm.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/rope.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/rope.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/sample.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/sample.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/scatter.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/scatter.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/sdpa.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/sdpa.air"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}/source/include" -c "${CMAKE_CURRENT_SOURCE_DIR}/source/topk.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/source/topk.air"
    COMMAND xcrun -sdk macosx metallib "${CMAKE_CURRENT_BINARY_DIR}/source/accumulate.air" "${CMAKE_CURRENT_BINARY_DIR}/source/convert.air" "${CMAKE_CURRENT_BINARY_DIR}/source/embeddings.air" "${CMAKE_CURRENT_BINARY_DIR}/source/expert_routing_metadata.air" "${CMAKE_CURRENT_BINARY_DIR}/source/gather_and_accumulate.air" "${CMAKE_CURRENT_BINARY_DIR}/source/matmul.air" "${CMAKE_CURRENT_BINARY_DIR}/source/moematmul.air" "${CMAKE_CURRENT_BINARY_DIR}/source/random.air" "${CMAKE_CURRENT_BINARY_DIR}/source/rmsnorm.air" "${CMAKE_CURRENT_BINARY_DIR}/source/rope.air" "${CMAKE_CURRENT_BINARY_DIR}/source/sample.air" "${CMAKE_CURRENT_BINARY_DIR}/source/scatter.air" "${CMAKE_CURRENT_BINARY_DIR}/source/sdpa.air" "${CMAKE_CURRENT_BINARY_DIR}/source/topk.air" -o "${METAL_LIB}"
    DEPENDS ${METAL_SOURCES}
    COMMENT "Compiling Metal compute library"
)

add_custom_target(build_metallib ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB})

add_library(log OBJECT source/log.c)

add_library(metal-kernels STATIC source/metal.m source/metal-kernels.c)
target_link_libraries(metal-kernels PRIVATE log)

add_dependencies(metal-kernels build_metallib)
add_custom_command(TARGET metal-kernels POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB}
            $<TARGET_FILE_DIR:metal-kernels>)

target_link_libraries(metal-kernels PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK} ${IOKIT_FRAMEWORK})

add_library(gptoss STATIC source/model.c source/tokenizer.c source/context.c)
target_link_libraries(gptoss PRIVATE log metal-kernels)

add_executable(generate source/generate.c)
target_link_libraries(generate gptoss)

# --- [ Tests
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(u32-random-test test/u32-random.cc)
target_link_libraries(u32-random-test PRIVATE GTest::gtest_main metal-kernels)
target_include_directories(u32-random-test PRIVATE source/include)
add_test(NAME u32-random-test COMMAND u32-random-test)

add_executable(f32-random-test test/f32-random.cc)
target_link_libraries(f32-random-test PRIVATE GTest::gtest_main metal-kernels)
target_include_directories(f32-random-test PRIVATE source/include)
add_test(NAME f32-random-test COMMAND f32-random-test)

add_executable(mf4-f32-convert-test test/mf4-f32-convert.cc)
target_link_libraries(mf4-f32-convert-test PRIVATE GTest::gtest_main metal-kernels)
target_include_directories(mf4-f32-convert-test PRIVATE source/include)
add_test(NAME mf4-f32-convert-test COMMAND mf4-f32-convert-test)

add_executable(bf16-f32-embeddings-test test/bf16-f32-embeddings.cc)
target_link_libraries(bf16-f32-embeddings-test PRIVATE GTest::gtest_main metal-kernels)
target_include_directories(bf16-f32-embeddings-test PRIVATE source/include)
add_test(NAME bf16-f32-embeddings-test COMMAND bf16-f32-embeddings-test)

add_executable(f32-bf16w-rmsnorm-test test/f32-bf16w-rmsnorm.cc)
target_link_libraries(f32-bf16w-rmsnorm-test PRIVATE GTest::gtest_main metal-kernels)
target_include_directories(f32-bf16w-rmsnorm-test PRIVATE source/include)
add_test(NAME f32-bf16w-rmsnorm-test COMMAND f32-bf16w-rmsnorm-test)

add_executable(f32-bf16w-matmul-test test/f32-bf16w-matmul.cc)
target_link_libraries(f32-bf16w-matmul-test PRIVATE GTest::gtest_main metal-kernels)
target_include_directories(f32-bf16w-matmul-test PRIVATE source/include)
add_test(NAME f32-bf16w-matmul-test COMMAND f32-bf16w-matmul-test)

add_executable(f32-rope-test test/f32-rope.cc)
target_link_libraries(f32-rope-test PRIVATE GTest::gtest_main metal-kernels)
target_include_directories(f32-rope-test PRIVATE source/include)
add_test(NAME f32-rope-test COMMAND f32-rope-test)

# --- [ Benchmarks
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable self-tests in Google Benchmark" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable installation of Google Benchmark" FORCE)
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.zip
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
)
FetchContent_MakeAvailable(benchmark)

add_executable(f32-random-bench benchmark/f32-random.cc)
target_link_libraries(f32-random-bench PRIVATE benchmark::benchmark metal-kernels)
target_include_directories(f32-random-bench PRIVATE source/include)

add_executable(u32-random-bench benchmark/u32-random.cc)
target_link_libraries(u32-random-bench PRIVATE benchmark::benchmark metal-kernels)
target_include_directories(u32-random-bench PRIVATE source/include)

add_executable(mf4-f32-convert-bench benchmark/mf4-f32-convert.cc)
target_link_libraries(mf4-f32-convert-bench PRIVATE benchmark::benchmark metal-kernels)
target_include_directories(mf4-f32-convert-bench PRIVATE source/include)

add_executable(f32-bf16w-rmsnorm-bench benchmark/f32-bf16w-rmsnorm.cc)
target_link_libraries(f32-bf16w-rmsnorm-bench PRIVATE benchmark::benchmark metal-kernels)
target_include_directories(f32-bf16w-rmsnorm-bench PRIVATE source/include)

add_executable(end-to-end-bench benchmark/end-to-end.cc)
target_link_libraries(end-to-end-bench PRIVATE benchmark::benchmark gptoss)
target_include_directories(end-to-end-bench PRIVATE source/include)

add_executable(end-to-end-threadgroup-bench benchmark/end-to-end-threadgroup.cc)
target_link_libraries(end-to-end-threadgroup-bench PRIVATE benchmark::benchmark gptoss)
target_include_directories(end-to-end-threadgroup-bench PRIVATE source/include)

# --- [ Python extension ] -----------------------------------------------
find_package(pybind11 CONFIG REQUIRED)          # provides pybind11_add_module

pybind11_add_module(_metal
    python/module.c
    python/context.c
    python/model.c
    python/tokenizer.c
)
set_target_properties(_metal PROPERTIES PREFIX "")

target_link_libraries(_metal PRIVATE gptoss)
add_dependencies(_metal build_metallib)
target_link_options(_metal PRIVATE
    LINKER:-sectcreate,__METAL,__shaders,${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB}
)
add_custom_command(TARGET _metal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB}
            $<TARGET_FILE_DIR:_metal>)

# 1️⃣  install the extension module into the Python package
install(TARGETS _metal LIBRARY DESTINATION gpt_oss/metal)

# 2️⃣  make sure the Metal shader archive travels with it
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB}
        DESTINATION gpt_oss/metal)
# ------------------------------------------------------------------------