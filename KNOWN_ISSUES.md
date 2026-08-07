# Known Issues

## Critical: Ndarray Shape Corruption Bug

**Status**: Under Investigation  
**Severity**: High  
**Affected**: `include/np/manipulation.hpp` - `diag()` function and possibly others

### Description
The `diag()` function exhibits undefined behavior where shape information becomes corrupted between function calls. Specifically:

1. Inside `diag()`, arrays are created with correct shape (e.g., [3,3])
2. After returning from `diag()`, the returned object reports correct shape
3. When calling `operator()(i, j)` on the returned object, the operator receives an object with corrupted shape (e.g., shape.size()=1 instead of 2)

### Debug Evidence
```
// Inside diag():
DEBUG result.ndim()=2, result.shape=[3,3]

// After returning:
d.shape.size()=2, d.ndim()=2, d.shape=[3,3], &d=0x...

// Inside operator()(i, j):
ERROR: operator()(i=0,j=0) called on object with shape.size()=1, ndim=1
  shape[0]=2
```

The same memory address reports different shapes, suggesting memory corruption or undefined behavior in how `Ndarray` manages its internal state.

### Root Cause Hypothesis
- Possible use-after-free in shape vector
- Stack corruption from incorrect buffer management
- Issue with how `Ndarray` constructor handles shape parameter
- Potential problem with shared_ptr usage in `data_` member

### Attempted Fixes
1. ✅ Fixed initializer_list confusion: `{size}` creates array with ONE element, not array of size `size`
2. ✅ Used `std::vector<int>{size, size}` → Still wrong (creates vector of `size` elements each = `size`)  
3. ✅ Used `std::vector<int> shape_vec = {size, size}` → Correct vector, but corruption still occurs

### Workaround
None currently. The `diag()` function and `test_manipulation` are non-functional.

### Next Steps
1. Review `Ndarray` class implementation for shape/data management bugs
2. Check for issues in copy constructor, move constructor, assignment operators
3. Verify strides calculation doesn't corrupt shape
4. Consider adding assertions to detect shape corruption earlier
5. Run with memory sanitizer (AddressSanitizer, Valgrind) to detect undefined behavior

### Related Files
- `include/np/ndarray.hpp` - Core Ndarray class
- `include/np/manipulation.hpp` - diag() function
- `tests/test_manipulation.cpp` - Failing test

---

## Minor: Narrowing Conversion Warnings in unique()

**Severity**: Low  
**File**: `include/np/manipulation.hpp`  
**Function**: `unique()`

Narrowing conversion warnings when creating result arrays:
```
warning: narrowing conversion of 'idx_size' from 'int' to 'long long unsigned int'
```

**Fix**: Cast to `std::size_t` explicitly or use `static_cast<std::size_t>(idx_size)`.

---

## Incomplete Implementations

### manipulation.hpp
- `insert()` - Only flat version implemented, axis version TODO
- `append()` - Only flat version implemented, axis version requires concatenate.hpp

### Missing High-Priority Functions
See `API_COVERAGE.md` for complete list. Key missing:
- `expand_dims()`, `atleast_1d/2d/3d()`, `broadcast_to()`
- `pad()` - Important!
- `meshgrid()`, `select()`, `choose()`, `place()`, `extract()`  
- `clip()`, `isclose()`, `allclose()`
- `cumsum()`, `cumprod()`, `diff()`

