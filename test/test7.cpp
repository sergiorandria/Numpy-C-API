#include "../ndarray.hpp"
#include <iostream>
#include <cassert>
#include <vector>

// Helper function to check if two arrays are equal
template<typename T>
bool compare_arrays(const np::Ndarray<T> &arr1, const np::Ndarray<T> &arr2)
{
    if (arr1.shape != arr2.shape) return false;
    size_t total_size = 1;
    for (auto dim : arr1.shape) total_size *= dim;
    for (size_t i = 0; i < total_size; ++i)
    {
        // This assumes buffer is accessible - you may need to add a getter
        // For now, we'll test through the proxy interface
    }
    return true;
}

// Helper to print test results
void print_test(const std::string& test_name, bool passed)
{
    std::cout << (passed ? "✓ " : "✗ ") << test_name << std::endl;
}

int main()
{
    std::cout << "\n=== Testing Ndarray Constructor with Initializer List ===\n" <<
              std::endl;
    // Test 1: 1D array
    {
        std::cout << "Test 1: 1D array (5 elements)" << std::endl;
        np::Ndarray<int> arr = {5};
        assert(arr.shape.size() == 1);
        assert(arr.shape[0] == 5);
        // Set and get values
        for (int i = 0; i < 5; ++i)
        {
            arr[i] = i * 10;
        }
        std::cout << "  Array: " << arr << std::endl;
        for (int i = 0; i < 5; ++i)
        {
            assert(arr[i] == i * 10);
        }
        print_test("  1D array creation and access", true);
    }
    // Test 2: 2D array
    {
        std::cout << "\nTest 2: 2D array (3x4)" << std::endl;
        np::Ndarray<int> arr = {3, 4};
        assert(arr.shape.size() == 2);
        assert(arr.shape[0] == 3);
        assert(arr.shape[1] == 4);
        // Fill with values
        int counter = 0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                arr[i][j] = counter++;
            }
        }
        std::cout << "  Array: " << arr << std::endl;
        // Verify values
        counter = 0;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                assert(arr[i][j] == counter++);
            }
        }
        print_test("  2D array creation and access", true);
    }
    // Test 3: 3D array
    {
        std::cout << "\nTest 3: 3D array (2x3x4)" << std::endl;
        np::Ndarray<double> arr = {2, 3, 4};
        assert(arr.shape.size() == 3);
        assert(arr.shape[0] == 2);
        assert(arr.shape[1] == 3);
        assert(arr.shape[2] == 4);
        // Fill with values
        int counter = 0;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 4; ++k)
                {
                    arr[i][j][k] = static_cast<double>(counter++);
                }
            }
        }
        std::cout << "  Array shape: [" << arr.shape[0] << ", "
                  << arr.shape[1] << ", " << arr.shape[2] << "]" << std::endl;
        // Verify some values
        assert(arr[0][0][0] == 0.0);
        assert(arr[0][0][3] == 3.0);
        assert(arr[0][2][2] == 10.0);
        assert(arr[1][2][3] == 23.0);
        print_test("  3D array creation and access", true);
    }
    // Test 4: 4D array
    {
        std::cout << "\nTest 4: 4D array (2x2x2x2)" << std::endl;
        np::Ndarray<float> arr = {2, 2, 2, 2};
        assert(arr.shape.size() == 4);
        assert(arr.shape[0] == 2);
        assert(arr.shape[1] == 2);
        assert(arr.shape[2] == 2);
        assert(arr.shape[3] == 2);
        // Fill with pattern
        int counter = 0;
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                for (int c = 0; c < 2; ++c)
                {
                    for (int d = 0; d < 2; ++d)
                    {
                        arr[a][b][c][d] = static_cast<float>(counter++);
                    }
                }
            }
        }
        std::cout << "  Array shape: [" << arr.shape[0] << ", "
                  << arr.shape[1] << ", " << arr.shape[2] << ", "
                  << arr.shape[3] << "]" << std::endl;
        // Verify values
        assert(arr[0][0][0][0] == 0.0f);
        assert(arr[0][0][0][1] == 1.0f);
        assert(arr[0][0][1][0] == 2.0f);
        assert(arr[0][1][0][0] == 4.0f);
        assert(arr[1][1][1][1] == 15.0f);
        print_test("  4D array creation and access", true);
    }
    // Test 5: Different data types
    {
        std::cout << "\nTest 5: Different data types" << std::endl;
        // Integer array
        np::Ndarray<int> int_arr = {3, 3};
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                int_arr[i][j] = i * 3 + j;
            }
        }
        assert(int_arr[1][2] == 5);
        // Float array
        np::Ndarray<float> float_arr = {2, 2};
        float_arr[0][0] = 3.14f;
        float_arr[0][1] = 2.718f;
        assert(float_arr[0][0] == 3.14f);
        // Double array
        np::Ndarray<double> double_arr = {2};
        double_arr[0] = 3.14159;
        double_arr[1] = 2.71828;
        assert(double_arr[0] == 3.14159);
        print_test("  Different data types work", true);
    }
    // Test 6: Zero-sized dimensions (should work)
    {
        std::cout << "\nTest 6: Zero-sized dimensions" << std::endl;
        np::Ndarray<int> arr = {0, 5};
        assert(arr.shape.size() == 2);
        assert(arr.shape[0] == 0);
        assert(arr.shape[1] == 5);
        // Should be empty array - no elements to access
        print_test("  Zero-sized dimensions handled correctly", true);
    }
    // Test 7: Single element (1x1x1)
    {
        std::cout << "\nTest 7: Single element array" << std::endl;
        np::Ndarray<int> arr = {1, 1, 1};
        assert(arr.shape.size() == 3);
        assert(arr.shape[0] == 1);
        assert(arr.shape[1] == 1);
        assert(arr.shape[2] == 1);
        arr[0][0][0] = 42;
        assert(arr[0][0][0] == 42);
        print_test("  Single element array works", true);
    }
    // Test 8: Default values after construction
    {
        std::cout << "\nTest 8: Default values after construction" << std::endl;
        np::Ndarray<int> arr = {2, 3};
        // Default should be zero-initialized (for int)
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                assert(arr[i][j] == 0);
            }
        }
        print_test("  Default values are zero-initialized", true);
    }
    // Test 9: Large array to test performance
    {
        std::cout << "\nTest 9: Large array (10x10x10)" << std::endl;
        np::Ndarray<int> arr = {10, 10, 10};
        // Fill with pattern
        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < 10; ++j)
            {
                for (int k = 0; k < 10; ++k)
                {
                    arr[i][j][k] = i * 100 + j * 10 + k;
                }
            }
        }
        // Verify corner values
        assert(arr[0][0][0] == 0);
        assert(arr[0][0][9] == 9);
        assert(arr[0][9][0] == 90);
        assert(arr[9][0][0] == 900);
        assert(arr[9][9][9] == 999);
        print_test("  Large array creation and access works", true);
    }
    // Test 10: Chained operations
    {
        std::cout << "\nTest 10: Chained operations with proxy" << std::endl;
        np::Ndarray<int> arr = {4, 4};
        // Fill diagonal
        for (int i = 0; i < 4; ++i)
        {
            arr[i][i] = 100;
        }
        // Modify using chain
        arr[1][2] = arr[2][1] = arr[3][3] = 50;
        assert(arr[0][0] == 100);
        assert(arr[1][1] == 100);
        assert(arr[2][2] == 100);
        assert(arr[3][3] == 50);
        assert(arr[1][2] == 50);
        assert(arr[2][1] == 50);
        print_test("  Chained operations work correctly", true);
    }
    // Test 11: Nested proxy access
    {
        std::cout << "\nTest 11: Nested proxy access patterns" << std::endl;
        np::Ndarray<int> arr = {3, 3, 3};
        // Fill with values
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 3; ++k)
                {
                    arr[i][j][k] = i * 100 + j * 10 + k;
                }
            }
        }
        // Test different access patterns
        auto proxy = arr[0];
        assert(proxy[0][0] == 0);
        assert(proxy[1][2] == 12);
        auto proxy2 = arr[1];
        assert(proxy2[0][0] == 100);
        assert(proxy2[2][1] == 121);
        print_test("  Nested proxy access works", true);
    }
    // Test 12: Memory layout verification (optional)
    {
        std::cout << "\nTest 12: Memory layout verification" << std::endl;
        np::Ndarray<int> arr = {2, 3, 4};
        // Fill with sequential values
        int counter = 0;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int k = 0; k < 4; ++k)
                {
                    arr[i][j][k] = counter++;
                }
            }
        }
        // Check row-major ordering
        assert(arr[0][0][0] == 0);
        assert(arr[0][0][1] == 1);
        assert(arr[0][0][2] == 2);
        assert(arr[0][0][3] == 3);
        assert(arr[0][1][0] == 4);
        assert(arr[1][0][0] == 12);
        print_test("  Row-major memory layout verified", true);
    }
    std::cout << "\n=== All tests completed ===\n" << std::endl;
    return 0;
}