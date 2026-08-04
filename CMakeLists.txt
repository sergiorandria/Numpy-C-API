cmake_minimum_required(VERSION 3.20)
project(np CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(-Wall -Wextra)
endif()

option(NP_COMPILED_UNITS
       "Enable the compile-time (negative-trait and constexpr-fold) test units"
       ON)

add_library(np INTERFACE)
add_library(np::np ALIAS np)
target_include_directories(np INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
