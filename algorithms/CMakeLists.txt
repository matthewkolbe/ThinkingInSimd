cmake_minimum_required (VERSION 3.16)

set(CMAKE_CXX_COMPILER clang++)

project ("algorithms")

add_compile_options(
  -Qunused-arguments
  -std=c++20
  -O3
  -march=native
)

set(SOURCES
    src/main.cpp
    src/avx_binary_search.cc
)

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING NO)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG origin/main
)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(algorithms ${SOURCES})

target_link_libraries(algorithms 
    benchmark::benchmark
)