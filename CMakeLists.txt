cmake_minimum_required(VERSION 3.29...4.0)

project(Simd
    LANGUAGES C CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FetchContent)

add_subdirectory("src")

option(SIMD_BUILD_EXAMPLES "Build example code" OFF)
if(SIMD_BUILD_EXAMPLES)
    add_subdirectory("example")
endif()
