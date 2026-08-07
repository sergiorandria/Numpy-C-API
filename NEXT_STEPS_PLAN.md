# Next Implementation Steps & API Documentation Plan

## Key Decisions Needed

### 1. numpy.array() vs Current `Ndarray` Class
**Question**: Should we implement `numpy.array()` as a free function or is `Ndarray<T>` constructor sufficient?

**NumPy Behavior**:
- `numpy.array()` is a factory function that creates ndarray objects
- Handles conversion from lists, tuples, other arrays
- Infers dtypes automatically
- Handles copy/no-copy semantics

**Current C++ Implementation**:
- `Ndarray<T>()` constructor - requires explicit type parameter
- `arange()`, `zeros()`, `ones()` - factory functions exist
- No generic `array()` function yet

**Recommendation**: 
- Keep `Ndarray<T>` as the main class
- Add `array()` as a convenience factory function for common cases
- Example: `auto a = np::array({1, 2, 3});` → deduces to `Ndarray<int>`

---

### 2. numpy.char.chararray - String Array Support
**Question**: Should we implement string/char array support?

**NumPy Behavior**:
- `numpy.char.chararray` - subclass of ndarray for string dtype
- Extensive string manipulation: capitalize, strip, split, join, etc.
- Both element-wise functions and array methods

**C++ Considerations**:
- C++ has `std::string` but no built-in string dtype concept like NumPy
- Would need `Ndarray<std::string>` specialization
- String operations would need element-wise implementations
- Large scope: 50+ string functions in numpy.char

**Recommendation**:
- **Defer for now** - Low priority for numerical computing library
- If needed later, implement as separate module `np::char::`
- Focus on core numerical operations first

---

### 3. __array_namespace_info__ - Array API Standard
**Question**: Should we implement array namespace info for standard compliance?

**NumPy Behavior**:
- `numpy.__array_namespace_info__()` - returns namespace info object
- Methods: `capabilities()`, `default_device()`, `default_dtypes()`, `devices()`, `dtypes()`
- Part of Python Array API standard for interoperability

**C++ Considerations**:
- No equivalent standard in C++
- Would be mostly compile-time information
- Could use C++20 concepts/traits instead

**Recommendation**:
- **Implement minimal version** for documentation purposes
- Create `np::array_namespace_info` namespace with:
  - Compile-time type traits for supported dtypes
  - Device info (currently: CPU only)
  - Capabilities list (what features are implemented)
- Use for generating API documentation

---

## 2. Private/Internal Function Naming Convention

### Current State
- No clear separation between public API and internal helpers
- `detail::` namespace exists for some internals
- No macro-based visibility control

### Proposed Convention

```cpp
// Option 1: Namespace-based (Current partial approach - RECOMMENDED)
namespace np {
    // Public API
    template <typename T>
    auto sum(const Ndarray<T>& arr) -> T;
    
    namespace detail {
        // Internal helpers - not for direct use
        template <typename T>
        auto compute_strides(const std::vector<int>& shape) -> std::vector<std::size_t>;
    }
}

// Option 2: Macro-based visibility (More explicit)
#define NP_API           // Public API (empty, for documentation)
#define NP_INTERNAL      // Internal API (empty, but marks intent)
#define NP_PRIVATE       // Private implementation detail

NP_API template <typename T>
auto sum(const Ndarray<T>& arr) -> T;

NP_INTERNAL template <typename T>
auto validate_axis(int axis, int ndim) -> int;
```

### Recommendation
**Use hybrid approach**:
1. **`detail::` namespace** for implementation helpers
2. **`NP_INTERNAL` macro** for functions that must be in public headers but shouldn't be called directly
3. **Doxygen `@internal` tag** in comments for documentation

```cpp
namespace np {
    /**
     * @brief Sum of array elements.
     * @public
     */
    template <typename T>
    auto sum(const Ndarray<T>& arr) -> T;
    
    /**
     * @internal
     * @brief Internal helper for axis validation.
     * Do not call directly - use in implementation only.
     */
    NP_INTERNAL inline auto validate_axis(int axis, int ndim) -> int {
        // implementation
    }
    
    namespace detail {
        // Pure implementation details
        template <typename T>
        auto compute_sum_kernel(const T* data, std::size_t size) -> T;
    }
}
```

---

## 3. API Documentation Strategy

### Create Comprehensive API Reference

**File Structure**:
```
docs/
├── API_REFERENCE.md         # Complete API listing
├── API_COVERAGE.md          # What's implemented vs NumPy
├── API_DIVERGENCE.md        # Known differences from NumPy
└── api/
    ├── array_creation.md
    ├── array_manipulation.md
    ├── mathematical.md
    ├── linear_algebra.md
    ├── random.md
    └── ...
```

**API_REFERENCE.md Structure**:
```markdown
# NumPy C++ API Reference

## Core Classes
- `Ndarray<T>` - Dynamic N-dimensional array
- `ndarray<T, Extents...>` - Compile-time fixed-size array
- `Matrix<T>` - 2D matrix (subclass of Ndarray)

## Array Creation
### From Scratch
- `zeros<T>(shape)` - Array filled with zeros
- `ones<T>(shape)` - Array filled with ones
- `empty<T>(shape)` - Uninitialized array
- `full<T>(shape, value)` - Array filled with value
- `arange<T>(start, stop, step)` - Evenly spaced values
- `linspace<T>(start, stop, num)` - Linear spacing
- `eye<T>(n, m, k)` - Identity matrix
...

### Status Indicators
✅ Fully implemented
⚠️ Partially implemented  
❌ Not implemented
🔄 Different from NumPy (see divergence docs)
```

### Generate from Code
Create a script to parse headers and generate documentation:

```python
# scripts/generate_api_docs.py
# Parses headers, extracts @brief, @param, etc.
# Generates markdown documentation
# Tracks implementation status
```

---

## 4. Priority Implementation Plan

### Phase 1: Core Missing Functions (High Priority)
These are commonly used and blocking users:

**Array Creation** (`creation.hpp`):
- ❌ `array()` - Generic array factory
- ❌ `asarray()`, `asanyarray()` - Array conversion
- ❌ `frombuffer()`, `fromfunction()` - Alternative constructors
- ❌ `meshgrid()`, `mgrid`, `ogrid` - Coordinate matrices

**Array Manipulation** (`manipulation.hpp` - additions):
- ❌ `expand_dims()` - Add dimension
- ❌ `squeeze()` - Remove single-dim axes (partially in ndarray)
- ❌ `atleast_1d()`, `atleast_2d()`, `atleast_3d()` - Dimension guarantees
- ❌ `broadcast_to()`, `broadcast_arrays()` - Broadcasting utilities
- ❌ `moveaxis()`, `rollaxis()` - Axis reordering
- ❌ `pad()` - Array padding (important!)
- ❌ `select()`, `choose()`, `place()`, `extract()` - Conditional ops

**Math Operations** (enhance `math.hpp`):
- ⚠️ More ufuncs: `clip()`, `sign()`, `copysign()`
- ❌ `around()`, `round()`, `fix()` - Rounding
- ❌ `gradient()` - Numerical gradient
- ❌ `diff()`, `ediff1d()` - Differences
- ❌ `trapezoid()`, `cumsum()`, `cumprod()` - Integration/accumulation

### Phase 2: Comparison & Logic (Medium Priority)
**Logic** (enhance `logic.hpp`):
- ⚠️ Element-wise comparisons - most exist
- ❌ `isclose()`, `allclose()` - Floating point comparison
- ❌ `array_equal()`, `array_equiv()` - Array comparison
- ❌ `isnan()`, `isinf()`, `isfinite()` - Special value checks

### Phase 3: Advanced Features (Lower Priority)
- Sorting & Searching
- Set Operations
- Polynomial functions (low priority)
- String arrays (numpy.char) - defer

### Phase 4: Array API Standard Compliance
- `__array_namespace_info__` minimal implementation
- Document deviations from standard
- Ensure naming consistency

---

## 5. Immediate Action Items

### A. Fix Current Issues
1. ✅ Fix NormOrd enum (DONE)
2. ⚠️ Debug test_manipulation heap corruption
3. ⚠️ Complete `insert()` axis implementation
4. ⚠️ Fix narrowing warnings in `unique()`

### B. Documentation
1. **Create API_REFERENCE.md** - Complete function listing
2. **Create API_COVERAGE.md** - Implementation status matrix
3. **Update AGENTS.md** - Add naming conventions
4. **Add inline documentation** - Ensure all public functions have Doxygen comments

### C. Code Organization
1. **Add NP_INTERNAL macro** to dtype.hpp, exceptions.hpp
2. **Review all public headers** - Mark internal functions
3. **Consolidate detail:: usage** - Move more helpers to detail namespace
4. **Create visibility guidelines** in AGENTS.md

### D. Testing Infrastructure
1. Fix heap corruption in test_manipulation
2. Add more edge case tests
3. Consider adding benchmark tests
4. Set up continuous integration if not already present

---

## 6. File Roadmap

```
include/np/
├── [Core - Implemented]
│   ├── np.hpp                 # Umbrella header
│   ├── ndarray.hpp            # Dynamic arrays ✅
│   ├── ndarray_fixed.hpp      # Fixed-size arrays ✅
│   ├── dtype.hpp              # Type system ✅
│   ├── exceptions.hpp         # Error types ✅
│   
├── [Creation - ~70% Complete]
│   ├── creation.hpp           # Basic creators ✅
│   ├── creation_fixed.hpp     # Fixed creators ✅
│   └── array_factory.hpp      # Generic array() - TODO
│   
├── [Manipulation - ~60% Complete]
│   ├── manipulation.hpp       # Implemented ✅
│   ├── indexing.hpp           # Advanced indexing - TODO
│   └── broadcasting.hpp       # Broadcast utils - TODO
│   
├── [Math - ~50% Complete]
│   ├── math.hpp               # Basic ufuncs ✅
│   ├── math_advanced.hpp      # More functions - TODO
│   └── accumulation.hpp       # cumsum, etc - TODO
│   
├── [Linear Algebra - ~80% Complete]
│   ├── linalg.hpp             # Core linalg ✅
│   ├── linalg_fixed.hpp       # Fixed linalg ✅
│   └── decomposition.hpp      # Advanced - partial
│   
├── [Other - Various Completion]
│   ├── logic.hpp              # Logical ops ~70% ✅
│   ├── random.hpp             # Random gen ✅
│   ├── fft.hpp                # FFT ✅
│   ├── matrix.hpp             # Matrix class ✅
│   ├── concatenate.hpp        # Concat ops ✅
│   ├── simd.hpp               # SIMD ✅
│   
└── [Planned - Not Started]
    ├── sorting.hpp            # Sort/search - TODO
    ├── set_ops.hpp            # Set operations - TODO
    ├── statistics.hpp         # Stats functions - TODO
    ├── comparison.hpp         # isclose, etc - TODO
    └── namespace_info.hpp     # Array API - TODO
```

---

## Summary of Recommendations

1. **numpy.array()**: Add as convenience factory, keep `Ndarray<T>` as primary
2. **numpy.char**: Defer - low priority for numerical library
3. **__array_namespace_info__**: Minimal compile-time implementation for documentation
4. **Naming**: Use `detail::` namespace + `NP_INTERNAL` macro + Doxygen tags
5. **Documentation**: Create comprehensive API_REFERENCE.md with status tracking
6. **Next Priority**: Fix current issues → Document API → Implement Phase 1 functions

## Next Session Goals
1. Create API documentation files
2. Add naming convention macros to AGENTS.md
3. Implement top 5 missing functions from Phase 1
4. Fix test_manipulation issues
