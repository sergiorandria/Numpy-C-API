# Array Manipulation Implementation Status

## Overview
Implemented comprehensive array manipulation functions in `include/np/manipulation.hpp`.

## Implemented Functions

### Rearranging Elements
- ✅ `flip()` - Reverse elements along axis
- ✅ `fliplr()` - Flip left-right (horizontal)
- ✅ `flipud()` - Flip up-down (vertical)  
- ✅ `roll()` - Roll array elements along axis
- ✅ `rot90()` - Rotate array by 90 degrees

### Tiling Arrays
- ✅ `tile()` - Construct array by repeating

### Building Matrices
- ✅ `diag()` - Extract or construct diagonal
- ✅ `diagflat()` - Create 2D array with flattened input on diagonal
- ✅ `tri()` - Lower triangular array
- ✅ `tril()` - Lower triangle of array
- ✅ `triu()` - Upper triangle of array
- ✅ `vander()` - Vandermonde matrix

### Splitting Arrays
- ✅ `split()` - Split array into multiple sub-arrays
- ✅ `array_split()` - Split into approximately equal pieces
- ✅ `hsplit()` - Split horizontally
- ✅ `vsplit()` - Split vertically
- ✅ `dsplit()` - Split along 3rd axis

### Adding/Removing Elements
- ✅ `delete_arr()` - Delete sub-arrays along axis
- ⚠️ `insert()` - Insert values (flat only, axis version not implemented)
- ⚠️ `append()` - Append values (flat only, requires concatenate.hpp for axis version)
- ✅ `trim_zeros()` - Trim leading/trailing zeros
- ✅ `unique()` - Find unique elements with optional indices/counts

### Conditional Selection
- ✅ `where()` - Select elements based on condition (both 1-arg and 3-arg forms)

## Files Created
- `include/np/manipulation.hpp` - Implementation (1130 lines)
- `tests/test_manipulation.cpp` - Test suite (275 lines)
- Updated `tests/CMakeLists.txt` to include test_manipulation

## Known Issues

### Fixed Issues
1. ✅ **Pre-existing compilation errors in `linalg.hpp`**: `NormOrd` enum not declared
   - Fixed by adding `NormOrd` enum declaration
   - Fixed missing `using R = real_t<T>` in `cond()` function
   
### Remaining Issues  
1. **Test execution issues**:
   - Heap corruption detected during test execution (Windows exception 0xc0000374)
   - Likely related to view/stride handling in `flip()` interacting with reshape
   - Needs debugging with sanitizers or simpler test cases

2. **Implementation Limitations**:
   - `insert()` - Only flat (no axis) version fully implemented
   - `append()` - Axis version requires `concatenate.hpp` to be included
   - Narrowing conversion warnings in `unique()` function (non-critical)

### Missing Functions (for future implementation)
- `expand_dims`, `atleast_1d`, `atleast_2d`, `atleast_3d`
- `broadcast_to`, `broadcast_arrays`
- `moveaxis`, `rollaxis`, `swapaxes` (swapaxes exists in ndarray.hpp)
- `pad` (important function)
- `select`, `choose`, `place`, `extract`
- `compress`
- `meshgrid`, `mgrid`, `ogrid`

## Next Steps
1. Fix pre-existing `NormOrd` compilation error in linalg.hpp
2. Debug heap corruption issue in test suite
3. Complete `insert()` axis version implementation
4. Implement remaining missing functions
5. Add manipulation.hpp to np.hpp (currently not auto-included per project convention)

## Testing Strategy
Once build issues are resolved:
1. Run individual function tests in isolation
2. Use address sanitizer to detect memory issues
3. Verify against NumPy reference documentation
4. Add edge case tests (empty arrays, single elements, etc.)

## References
- NumPy array manipulation routines: `numpy-reference/reference/routines.array-manipulation.html`
- Individual function docs in `numpy-reference/reference/generated/numpy.*.html`
