# Disable the Google Benchmark requirement on Google Test
include(FetchContent)

FetchContent_Declare(
    vectorclass
    GIT_REPOSITORY https://github.com/vectorclass/version2.git
    SOURCE_DIR "vectorclass"
)

FetchContent_MakeAvailable(vectorclass)

add_library(vectorclass INTERFACE)
target_include_directories(vectorclass INTERFACE ${vectorclass_SOURCE_DIR})