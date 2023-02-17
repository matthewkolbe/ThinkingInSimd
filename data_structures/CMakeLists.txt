cmake_minimum_required (VERSION 3.16)

set(CMAKE_CXX_COMPILER clang++)

project ("data_structures")

add_compile_options(
  -Qunused-arguments
  -std=c++20
  -O3
  -march=native
  -fopenmp
  -ffast-math
  -lm
  -lstdc++
  -lpthread
)

set(SOURCES
    src/main.cpp
    src/black_scholes.cc
    src/vec_black_scholes.cc
    src/data_structures.cc
    src/constants.cc
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

add_executable(data_structures ${SOURCES})

find_package(OpenMP REQUIRED)
target_link_libraries(data_structures 
    benchmark::benchmark
    OpenMP::OpenMP_CXX
)