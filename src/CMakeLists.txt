# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.16)

project ("ThinkingInSimd")

add_compile_options(
  -std=c++20
  -O3
  -march=native
  -fopenmp
  -ffast-math
  -lm
  -lstdc++
  -lbenchmark
  -lpthread
)

set(SOURCES
    benchmarks/main.cpp
    benchmarks/black_scholes.cc
    benchmarks/vec_black_scholes.cc
)

include(FetchContent)

FetchContent_Declare(vectorclass
  GIT_REPOSITORY https://github.com/vectorclass/version2.git
)

set(BENCHMARK_ENABLE_TESTING NO)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG origin/main
)

FetchContent_MakeAvailable(vectorclass googlebenchmark)

include_directories(${vectorclass_SOURCE_DIR})


# Create the executable
add_executable(ThinkingInSimd ${SOURCES})

target_link_libraries(ThinkingInSimd benchmark::benchmark)