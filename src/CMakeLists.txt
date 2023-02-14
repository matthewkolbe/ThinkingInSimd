# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.11)

project ("ThinkingInSimd")

set( CMAKE_CXX_STANDARD 20)
set( CMAKE_CXX_STANDARD_REQUIRED YES )
set( CMAKE_CXX_FLAGS "-O3 -lm -lstdc++ -march=native -fopenmp -ffast-math  -lbenchmark -lpthread")

set(SOURCES
    src/benchmarks/main.cpp
    src/benchmarks/black_scholes.cc
    src/benchmarks/vec_black_scholes.cc
)

include_directories(
    _deps/vectorclass-src/vectorclass
)

# Link Google Benchmark to the project
target_link_libraries(ThinkingInSimd benchmark::benchmark)

# Create the executable
add_executable(ThinkingInSimd ${SOURCES})


# Include sub-projects.
add_subdirectory ("benchmarks")
