// test_matrix.cpp
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <array>
#include "../Matrix/Matrix.h"

namespace np
{

// Helper function to compare matrices with tolerance for floating point
    template<typename T>
    bool matrices_equal(const Matrix<T> &a, const Matrix<T> &b,
                        T tolerance = T(1e-6))
    {
        if (a.rows() != b.rows() || a.cols() != b.cols())
        {
            return false;
        }
        for (size_t i = 0; i < a.rows(); ++i)
        {
            for (size_t j = 0; j < a.cols(); ++j)
            {
                T diff = a(i, j) - b(i, j);
                if (diff < -tolerance || diff > tolerance)
                {
                    std::cout << "Mismatch at (" << i << "," << j << "): "
                              << a(i, j) << " vs " << b(i, j) << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

// Helper to print test results
    void print_test_result(const std::string& test_name, bool passed)
    {
        std::cout << (passed ? "✓ " : "✗ ") << test_name << std::endl;
    }

} // namespace np

int main()
{
    using namespace np;
    std::cout << "=== Matrix Class Tests ===\n\n";
    int tests_passed = 0;
    int total_tests = 0;
    // Test 1: Construction from 2D vector
    {
        std::cout << "Test 1: Construction from 2D vector\n";
        std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        Matrix<int> mat(data);
        bool passed = (mat.rows() == 3 && mat.cols() == 3);
        if (passed)
        {
            passed = (mat(0, 0) == 1 && mat(0, 1) == 2 && mat(0, 2) == 3 &&
                      mat(1, 0) == 4 && mat(1, 1) == 5 && mat(1, 2) == 6 &&
                      mat(2, 0) == 7 && mat(2, 1) == 8 && mat(2, 2) == 9);
        }
        print_test_result("2D vector construction", passed);
        if (passed) tests_passed++;
        total_tests++;
        mat.print();
        std::cout << std::endl;
    }
    // Test 2: Construction from 1D vector (column vector)
    {
        std::cout << "Test 2: Construction from 1D vector (column vector)\n";
        std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
        Matrix<double> mat(data);
        bool passed = (mat.rows() == 4 && mat.cols() == 1);
        if (passed)
        {
            passed = (mat(0, 0) == 1.0 && mat(1, 0) == 2.0 &&
                      mat(2, 0) == 3.0 && mat(3, 0) == 4.0);
        }
        print_test_result("1D vector to column vector", passed);
        if (passed) tests_passed++;
        total_tests++;
        mat.print();
        std::cout << std::endl;
    }
    // Test 3: Construction with dimensions
    {
        std::cout << "Test 3: Construction with dimensions\n";
        Matrix<int> mat(2, 3, 42);
        bool passed = (mat.rows() == 2 && mat.cols() == 3);
        if (passed)
        {
            for (size_t i = 0; i < 2; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    if (mat(i, j) != 42)
                    {
                        passed = false;
                        break;
                    }
                }
            }
        }
        print_test_result("Dimension construction with initial value", passed);
        if (passed) tests_passed++;
        total_tests++;
        mat.print();
        std::cout << std::endl;
    }
    // Test 4: String construction
    {
        std::cout << "Test 4: Construction from string\n";
        std::string str = "hello";
        Matrix<char> mat(str);
        bool passed = (mat.rows() == 1 && mat.cols() == 5);
        if (passed)
        {
            passed = (mat(0, 0) == 'h' && mat(0, 1) == 'e' &&
                      mat(0, 2) == 'l' && mat(0, 3) == 'l' && mat(0, 4) == 'o');
        }
        print_test_result("String to character matrix", passed);
        if (passed) tests_passed++;
        total_tests++;
        mat.print();
        std::cout << std::endl;
    }
    // Test 5: Matrix multiplication
    {
        std::cout << "Test 5: Matrix multiplication\n";
        Matrix<int> a({{1, 2}, {3, 4}});
        Matrix<int> b({{5, 6}, {7, 8}});
        Matrix<int> expected({{19, 22}, {43, 50}});
        Matrix<int> result = a * b;
        bool passed = matrices_equal(result, expected);
        print_test_result("Matrix multiplication (2x2)", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << "A:\n"; a.print();
        std::cout << "B:\n"; b.print();
        std::cout << "A * B:\n"; result.print();
        std::cout << std::endl;
    }
    // Test 6: Matrix addition
    {
        std::cout << "Test 6: Matrix addition\n";
        Matrix<int> a({{1, 2}, {3, 4}});
        Matrix<int> b({{5, 6}, {7, 8}});
        Matrix<int> expected({{6, 8}, {10, 12}});
        Matrix<int> result = a + b;
        bool passed = matrices_equal(result, expected);
        print_test_result("Matrix addition", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << "A + B:\n"; result.print();
        std::cout << std::endl;
    }
    // Test 7: Scalar multiplication
    {
        std::cout << "Test 7: Scalar multiplication\n";
        Matrix<int> mat({{1, 2}, {3, 4}});
        Matrix<int> expected({{2, 4}, {6, 8}});
        Matrix<int> result = mat * 2;
        bool passed = matrices_equal(result, expected);
        print_test_result("Scalar multiplication", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << "Mat * 2:\n"; result.print();
        std::cout << std::endl;
    }
    // Test 8: Transpose
    {
        std::cout << "Test 8: Transpose\n";
        Matrix<int> mat({{1, 2, 3}, {4, 5, 6}});
        Matrix<int> expected({{1, 4}, {2, 5}, {3, 6}});
        Matrix<int> result = mat.transpose();
        bool passed = (result.rows() == 3 && result.cols() == 2);
        if (passed)
        {
            passed = matrices_equal(result, expected);
        }
        print_test_result("Matrix transpose", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << "Original (2x3):\n"; mat.print();
        std::cout << "Transpose (3x2):\n"; result.print();
        std::cout << std::endl;
    }
    // Test 9: Square matrix detection
    {
        std::cout << "Test 9: Square matrix detection\n";
        Matrix<int> square({{1, 2}, {3, 4}});
        Matrix<int> non_square({{1, 2, 3}, {4, 5, 6}});
        bool passed = (square.is_square() && !non_square.is_square());
        print_test_result("is_square() method", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << "Square matrix (2x2): " << (square.is_square() ? "square" :
                  "not square") << std::endl;
        std::cout << "Non-square matrix (2x3): " << (non_square.is_square() ? "square" :
                  "not square") << std::endl;
        std::cout << std::endl;
    }
    // Test 10: Out of bounds access
    {
        std::cout << "Test 10: Out of bounds access exception\n";
        Matrix<int> mat(2, 2);
        bool passed = false;
        try
        {
            mat(5, 5) = 42;  // Should throw
        }
        catch (const std::out_of_range &)
        {
            passed = true;
        }
        print_test_result("Out of bounds throws exception", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << std::endl;
    }
    // Test 11: Matrix from Ndarray
    {
        std::cout << "Test 11: Construction from Ndarray\n";
        Ndarray<int> arr({2, 2});
        arr[0][0] = 1;
        arr[0][1] = 2;
        arr[1][0] = 3;
        arr[1][1] = 4;
        Matrix<int> mat(arr);
        bool passed = (mat.rows() == 2 && mat.cols() == 2 &&
                       mat(0, 0) == 1 && mat(0, 1) == 2 &&
                       mat(1, 0) == 3 && mat(1, 1) == 4);
        print_test_result("Construction from Ndarray", passed);
        if (passed) tests_passed++;
        total_tests++;
        mat.print();
        std::cout << std::endl;
    }
    // Test 12: In-place operations
    {
        std::cout << "Test 12: In-place operations\n";
        Matrix<int> a({{1, 2}, {3, 4}});
        Matrix<int> b({{5, 6}, {7, 8}});
        Matrix<int> expected({{6, 8}, {10, 12}});
        a += b;
        bool passed = matrices_equal(a, expected);
        print_test_result("In-place addition (+=)", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << "After a += b:\n"; a.print();
        std::cout << std::endl;
    }
    // Test 13: Matrix multiplication with vector
    {
        std::cout << "Test 13: Matrix * vector multiplication\n";
        Matrix<int> mat({{1, 2}, {3, 4}});
        Ndarray<int> vec({2, 1});
        vec[0][0] = 5;
        vec[1][0] = 6;
        Ndarray<int> expected({2, 1});
        expected[0][0] = 17; // 1*5 + 2*6
        expected[1][0] = 39; // 3*5 + 4*6
        Ndarray<int> result = mat * vec;
        bool passed = (result[0][0] == 17 && result[1][0] == 39);
        print_test_result("Matrix-vector multiplication", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << "Result:\n";
        for (size_t i = 0; i < result.shape[0]; ++i)
        {
            std::cout << "[" << result[i][0] << "]\n";
        }
        std::cout << std::endl;
    }
    // Test 14: Type deduction
    {
        std::cout << "Test 14: Automatic type deduction\n";
        std::vector<std::vector<int>> data = {{1, 2}, {3, 4}};
        Matrix mat(data);  // Should deduce Matrix<int>
        bool passed = (std::is_same_v<decltype(mat), Matrix<int>>);
        print_test_result("CTAD works", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << std::endl;
    }
    // Test 15: Copy and move semantics
    {
        std::cout << "Test 15: Copy and move semantics\n";
        Matrix<int> original({{1, 2}, {3, 4}});
        Matrix<int> copy = original;
        Matrix<int> moved = std::move(original);
        bool passed = (copy(0, 0) == 1 && copy(0, 1) == 2 &&
                       copy(1, 0) == 3 && copy(1, 1) == 4 &&
                       moved(0, 0) == 1 && moved(0, 1) == 2 &&
                       moved(1, 0) == 3 && moved(1, 1) == 4);
        print_test_result("Copy and move constructors", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << std::endl;
    }
    // Test 16: Large matrix multiplication (performance test)
    {
        std::cout << "Test 16: Large matrix multiplication\n";
        Matrix<int> a(10, 10, 2);
        Matrix<int> b(10, 10, 3);
        bool passed = true;
        try
        {
            Matrix<int> result = a * b;
            passed = (result.rows() == 10 && result.cols() == 10);
            // Check a few values: each element should be 2*3*10 = 60
            for (int i = 0; i < 10 && passed; ++i)
            {
                for (int j = 0; j < 10 && passed; ++j)
                {
                    if (result(i, j) != 60)
                    {
                        passed = false;
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            passed = false;
            std::cout << "Exception: " << e.what() << std::endl;
        }
        print_test_result("10x10 matrix multiplication", passed);
        if (passed) tests_passed++;
        total_tests++;
        std::cout << std::endl;
    }
    // Summary
    std::cout << "=== Test Summary ===\n";
    std::cout << "Passed: " << tests_passed << "/" << total_tests << "\n";
    std::cout << "Success rate: " << (100.0 * tests_passed / total_tests) << "%\n";
    if (tests_passed == total_tests)
    {
        std::cout << "\n🎉 All tests passed!\n";
        return 0;
    }
    else
    {
        std::cout << "\n❌ Some tests failed!\n";
        return 1;
    }
}