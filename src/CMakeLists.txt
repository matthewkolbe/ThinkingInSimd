# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.11)

project ("ThinkingInSimd")

set(SOURCES
    benchmarks/main.cpp
    benchmarks/black_scholes.cc
    benchmarks/vec_black_scholes.cc
)

include_directories(${PROJECT_SOURCE_DIR}/_deps/vectorclass-src)

# Create the executable
add_executable(ThinkingInSimd ${SOURCES})

# Link Google Benchmark to the project
target_link_libraries(ThinkingInSimd benchmark::benchmark)

# Include sub-projects.
add_subdirectory ("benchmarks")

# Compiler flags
target_compile_options(ThinkingInSimd PRIVATE
  -std=c++20
  -O3
  -march=native
  -fopenmp
  -ffast-math
)

target_link_libraries(ThinkingInSimd
  -lm
  -lstdc++
  -lbenchmark
  -lpthread
)