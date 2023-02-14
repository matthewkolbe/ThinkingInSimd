cmake_minimum_required (VERSION 3.11)

# Include Google Benchmark
include(../cmake/GoogleBenchmark.cmake)
include(../cmake/VectorClass.cmake)

# Add source to this project's executable.
add_executable (ThinkingInSimd "main.cpp" "black_scholes.cc" "vec_black_scholes.cc")

# Link Google Benchmark to the project
target_link_libraries(ThinkingInSimd benchmark::benchmark)

target_link_libraries(ThinkingInSimd vectorclass)