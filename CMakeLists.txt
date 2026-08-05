cmake_minimum_required(VERSION 3.20)
project(np CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(-Wall -Wextra)
endif()

# SIMD optimization options
option(NP_ENABLE_SIMD "Enable SIMD optimizations (SSE2/AVX/AVX2/AVX-512/NEON)" ON)
option(NP_ENABLE_AVX2 "Enable AVX2 instructions (implies AVX)" OFF)
option(NP_ENABLE_AVX512 "Enable AVX-512 instructions (implies AVX2)" OFF)

if(NP_ENABLE_SIMD)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        # x86/x86-64 architecture
        if(MSVC)
            # MSVC compiler flags
            if(NP_ENABLE_AVX512)
                add_compile_options(/arch:AVX512)
            elseif(NP_ENABLE_AVX2)
                add_compile_options(/arch:AVX2)
            else()
                # SSE2 is enabled by default on x64
                add_compile_options(/arch:SSE2)
            endif()
        else()
            # GCC/Clang compiler flags
            if(NP_ENABLE_AVX512)
                add_compile_options(-mavx512f -mavx512dq)
            elseif(NP_ENABLE_AVX2)
                add_compile_options(-mavx2 -mfma)
            else()
                # Enable SSE4.2 by default for better performance
                add_compile_options(-msse4.2)
            endif()
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        # ARM64 architecture - NEON is standard
        if(NOT MSVC)
            add_compile_options(-march=armv8-a)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        # ARM32 architecture
        if(NOT MSVC)
            add_compile_options(-mfpu=neon -mfloat-abi=hard)
        endif()
    endif()
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
