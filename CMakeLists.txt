cmake_minimum_required(VERSION 3.20)
project(numpy-cpp
  VERSION 2.2.0
  DESCRIPTION "Header-only C++20 NumPy 2.2 — 712 routines, zero Python runtime"
  HOMEPAGE_URL "https://github.com/sergiorandria/numpy-cpp"
  LANGUAGES CXX)

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
option(NP_ENABLE_PQC "Enable post-quantum constant-time hardening (pqc.hpp)" ON)
option(NP_ENABLE_RVV "Enable RISC-V Vector (RVV 1.0) when targeting riscv64" OFF)
option(NP_ENABLE_WASM_SIMD "Enable WASM SIMD128 when targeting wasm32/wasm64" OFF)
option(NP_ENABLE_SVE "Enable ARM SVE/SVE2 when targeting aarch64" OFF)
option(NP_ENABLE_VSX "Enable POWER VSX when targeting powerpc64" OFF)
set(NP_PQC_ALG "" CACHE STRING "PQC algorithm for __NUMPY_PQC_ALG (e.g., MLKEM768, MLDSA65) – empty = none")

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
if(NP_PQC_ALG)
    add_compile_definitions(__NUMPY_PQC_ALG=${NP_PQC_ALG})
endif()

# ── Bigint / GMP (header-only cpp_int, optional GMP mpz) ─────────────────
option(NP_ENABLE_BIGINT "Enable bigint support (boost::multiprecision::cpp_int)" ON)
option(NP_ENABLE_GMP "Enable GMP mpz backend for bigint (requires libgmp)" OFF)
if(NP_ENABLE_GMP)
    find_library(GMP_LIB gmp)
    find_path(GMP_INCLUDE_DIR gmp.h)
    if(GMP_LIB AND GMP_INCLUDE_DIR)
        message(STATUS "Found GMP: ${GMP_LIB}")
    else()
        message(WARNING "NP_ENABLE_GMP=ON but libgmp not found; falling back to cpp_int")
    endif()
endif()

option(NP_COMPILED_UNITS
       "Enable the compile-time (negative-trait and constexpr-fold) test units"
       ON)

option(NP_USE_THREADING "Enable threadpool for sorting/searching/loops (NP_THREADPOOL)" OFF)

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

add_library(numpy-cpp INTERFACE)
add_library(numpy-cpp::numpy-cpp ALIAS numpy-cpp)
target_include_directories(numpy-cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
find_package(Threads REQUIRED)
target_link_libraries(numpy-cpp INTERFACE Threads::Threads)
if(NP_ENABLE_GMP AND GMP_LIB)
    target_link_libraries(numpy-cpp INTERFACE ${GMP_LIB})
    target_include_directories(numpy-cpp INTERFACE ${GMP_INCLUDE_DIR})
    target_compile_definitions(numpy-cpp INTERFACE NP_HAS_GMP=1 NP_HAS_GMP_H=1)
endif()
if(NP_ENABLE_BIGINT)
    target_compile_definitions(numpy-cpp INTERFACE NP_HAS_BIGINT=1)
    find_package(Boost QUIET)
    if(Boost_FOUND)
        target_include_directories(numpy-cpp INTERFACE ${Boost_INCLUDE_DIRS})
    endif()
endif()
# ── LLVM JIT for differential VM (optional, header-only fallback) ─────
# No find_package needed: differential.hpp uses __has_include(<llvm/IR/IRBuilder.h>)
# to enable JIT when LLVM headers are present. No link required for fallback.
option(NP_ENABLE_LLVM "Enable LLVM JIT for differential VM (requires LLVM dev headers)" OFF)
if(NP_ENABLE_LLVM)
  target_compile_definitions(numpy-cpp INTERFACE NP_ENABLE_LLVM=1)
  find_package(LLVM QUIET)
  if(LLVM_FOUND)
    target_include_directories(numpy-cpp INTERFACE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(numpy-cpp INTERFACE LLVM)
  endif()
endif()
if(NP_USE_THREADING)
  target_compile_definitions(numpy-cpp INTERFACE NP_USE_THREADING)
endif()

# ── Legacy alias: keep `np::np` working for existing downstream code ──
add_library(np INTERFACE)
target_link_libraries(np INTERFACE numpy-cpp)
add_library(np::np ALIAS np)

# ── Install & package (header-only) ─────────────────────────────────────
include(GNUInstallDirs)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS numpy-cpp np
        EXPORT numpy-cppTargets)
install(EXPORT numpy-cppTargets
        FILE numpy-cppTargets.cmake
        NAMESPACE numpy-cpp::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/numpy-cpp")
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/numpy-cppConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/numpy-cppConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/numpy-cppConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/numpy-cpp")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/numpy-cppConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/numpy-cppConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/numpy-cpp")

include(CTest)
# Only build tests when this is the top-level project (avoids building tests
# for every FetchContent / add_subdirectory consumer).
if(BUILD_TESTING AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(tests)
endif()

option(NP_BUILD_PYTHON "Build Python bridge via pybind11 (header-only, optional)" OFF)
if(NP_BUILD_PYTHON)
  add_subdirectory(python)
endif()
