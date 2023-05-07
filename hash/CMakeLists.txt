cmake_minimum_required (VERSION 3.16)

set(CMAKE_CXX_COMPILER clang-16)

project ("hash")

add_compile_options(
  -Qunused-arguments
  -std=c++20
  -O3
  -march=native
)

set(SOURCES
    src/main.cpp
    src/fash.hh
)

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING NO)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG origin/main
)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(hash ${SOURCES})

target_link_libraries(hash 
    benchmark::benchmark
)