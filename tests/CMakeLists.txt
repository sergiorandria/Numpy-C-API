set(NP_TESTS
    test_dtype
    test_proxy
    test_ndarray
    test_creation
    test_linalg
    test_matrix
    test_fft
    test_fixed
    test_math
    test_logic
    test_concatenate
    test_random
    test_simd
    test_manipulation
    test_char
    test_chararray
    test_unimplemented)

if(NP_COMPILED_UNITS)
    list(APPEND NP_TESTS test_constexpr test_compile_time)
endif()

foreach(t ${NP_TESTS})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE np::np)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
