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
        # ARM64 architecture - NEON is standard, SVE optional
        if(NOT MSVC)
            if(NP_ENABLE_SVE)
                add_compile_options(-march=armv8-a+sve -msve-vector-bits=scalable)
            else()
                add_compile_options(-march=armv8-a)
            endif()
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
        # ARM32 architecture
        if(NOT MSVC)
            add_compile_options(-mfpu=neon -mfloat-abi=hard)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv64|riscv32)")
        if(NOT MSVC AND NP_ENABLE_RVV)
            add_compile_options(-march=rv64gcv -mabi=lp64d)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(wasm32|wasm64)")
        if(NOT MSVC AND NP_ENABLE_WASM_SIMD)
            add_compile_options(-msimd128)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64|powerpc64)")
        if(NOT MSVC AND NP_ENABLE_VSX)
            add_compile_options(-mcpu=power9 -mvsx)
        endif()
    endif()
endif()
if(NP_ENABLE_PQC)
    add_compile_definitions(NP_ENABLE_PQC=1)
endif()

option(NP_COMPILED_UNITS
       "Enable the compile-time (negative-trait and constexpr-fold) test units"
       ON)

option(NP_USE_THREADING "Enable threadpool for sorting/searching/loops (NP_THREADPOOL)" OFF)

option(NP_ENABLE_PQC "Enable post-quantum constant-time hardening (pqc.hpp)" ON)
option(NP_ENABLE_RVV "Enable RISC-V Vector (RVV 1.0) when targeting riscv64" OFF)
option(NP_ENABLE_WASM_SIMD "Enable WASM SIMD128 when targeting wasm32/wasm64" OFF)
option(NP_ENABLE_SVE "Enable ARM SVE/SVE2 when targeting aarch64" OFF)
option(NP_ENABLE_VSX "Enable POWER VSX when targeting powerpc64" OFF)

# ── Compilation optimization ──────────────────────────────────────────
# Tuned for powerful multi-core machines; all options are OFF by default
# to keep debug builds fast, but can be enabled for Release.
option(NP_ENABLE_LTO "Enable Link-Time Optimization (IPO/LTCG)" OFF)
option(NP_ENABLE_NATIVE "Enable -march=native / -mtune=native (GCC/Clang) / /arch:AVX* tuning" OFF)
option(NP_ENABLE_FAST_MATH "Enable -ffast-math / /fp:fast (may affect NaN/inf)" OFF)
option(NP_ENABLE_O3 "Force -O3 / /O2 for Release even if not default" ON)

if(NP_ENABLE_O3)
  if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
  else()
    add_compile_options($<$<CONFIG:Release>:-O3> $<$<CONFIG:RelWithDebInfo>:-O3>)
  endif()
endif()

if(NP_ENABLE_FAST_MATH)
  if(MSVC)
    add_compile_options($<$<CONFIG:Release>:/fp:fast> $<$<CONFIG:RelWithDebInfo>:/fp:fast>)
  else()
    add_compile_options($<$<CONFIG:Release>:-ffast-math> $<$<CONFIG:RelWithDebInfo>:-ffast-math>)
  endif()
endif()

if(NP_ENABLE_NATIVE)
  if(MSVC)
    # MSVC has no direct -march=native; keep AVX tuning from SIMD block
    # and add /Ot for speed. User can set /arch:AVX2 etc. via NP_ENABLE_AVX2.
    add_compile_options($<$<CONFIG:Release>:/Ot>)
  else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
      add_compile_options($<$<CONFIG:Release>:-march=native> $<$<CONFIG:RelWithDebInfo>:-march=native>)
    else()
      add_compile_options($<$<CONFIG:Release>:-mtune=native> $<$<CONFIG:RelWithDebInfo>:-mtune=native>)
    endif()
  endif()
endif()

if(NP_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _np_ipo_ok OUTPUT _np_ipo_msg)
  if(_np_ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    if(MSVC)
      # MSVC LTCG needs /GL + /LTCG; CMake handles via IPO flag
      add_compile_options($<$<CONFIG:Release>:/GL> $<$<CONFIG:RelWithDebInfo>:/GL>)
    else()
      add_compile_options($<$<CONFIG:Release>:-flto> $<$<CONFIG:RelWithDebInfo>:-flto>)
      add_link_options($<$<CONFIG:Release>:-flto> $<$<CONFIG:RelWithDebInfo>:-flto>)
    endif()
  else()
    message(WARNING "LTO requested but not supported: ${_np_ipo_msg}")
  endif()
endif()

# Release: always define NDEBUG and enable function/data sections + GC for
# smaller/faster binaries (GCC/Clang). Also enable -pipe for faster compilation.
if(NOT MSVC)
  add_compile_options($<$<CONFIG:Release>:-DNDEBUG> $<$<CONFIG:RelWithDebInfo>:-DNDEBUG>)
  add_compile_options($<$<CONFIG:Release>:-ffunction-sections> $<$<CONFIG:Release>:-fdata-sections>)
  add_link_options($<$<CONFIG:Release>:-Wl,--gc-sections>)
  add_compile_options($<$<CONFIG:Release>:-pipe> $<$<CONFIG:RelWithDebInfo>:-pipe>)
else()
  add_compile_definitions($<$<CONFIG:Release>:NDEBUG> $<$<CONFIG:RelWithDebInfo>:NDEBUG>)
endif()

# Adaptive parallelism for compilation itself (ninja/make -j)
# If building on powerful machine, allow more jobs via CMAKE_BUILD_PARALLEL_LEVEL
# (no flag needed – CMake respects -j).

add_library(np INTERFACE)
add_library(np::np ALIAS np)
target_include_directories(np INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
find_package(Threads REQUIRED)
target_link_libraries(np INTERFACE Threads::Threads)
if(NP_USE_THREADING)
  target_compile_definitions(np INTERFACE NP_USE_THREADING)
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
