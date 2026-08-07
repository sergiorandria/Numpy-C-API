# NumPy char Module - Professional Refactoring Plan

## Current Issues

1. **Section dividers**: Using `====` style comments (non-professional)
2. **No file header**: Missing author, copyright, brief description
3. **Monolithic structure**: All functions inline without helper organization
4. **No detail namespace**: Helper functions mixed with public API
5. **Inconsistent spacing**: Not following project conventions
6. **Missing error contexts**: Generic error messages
7. **No const correctness**: Some functions could use const ref parameters

## Professional Structure (Following linalg.hpp Pattern)

### 1. File Header
```cpp
/**
 * @file char.hpp
 * @brief String operations for arrays of std::string (numpy.char module).
 *
 * Implements element-wise string manipulation functions matching numpy.char
 * semantics. All functions operate on Ndarray<std::string> and return either
 * string arrays, boolean arrays, or integer arrays depending on the operation.
 *
 * Function signatures, parameter order, and default values mirror the numpy.char
 * pages in numpy-reference/reference/generated/numpy.char.*.
 *
 * Implementation notes:
 *  - encode() and decode() are no-ops (C++ std::string is already byte-based)
 *  - isdecimal() and isnumeric() use isdigit() approximation
 *  - split(), rsplit(), splitlines() return flattened arrays
 *  - translate() uses simplified 256-character table
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
```

### 2. Namespace Organization
```cpp
namespace np {
namespace ch {

    /* Internal helpers - character classification and string utilities */
    namespace detail {
        // Helper functions here
        inline bool str_islower(const std::string& s);
        inline bool str_isupper(const std::string& s);
        inline bool str_istitle(const std::string& s);
       
        inline void validate_shapes(const Ndarray<std::string>& a,
                                     const Ndarray<std::string>& b,
                                     const char* func_name);
    }

    /* String operations - addition, repetition, formatting */
    // Public API functions

    /* Case conversion operations */
    // capitalize, lower, upper, swapcase, title

    /* Padding and alignment operations */
    // center, ljust, rjust, zfill

    /* Trimming operations */
    // strip, lstrip, rstrip

    /* String manipulation operations */
    // replace, expandtabs

    /* Comparison operations */
    // equal, not_equal, less, greater, etc.

    /* Information and search operations */
    // count, find, rfind, index, rindex, str_len, startswith, endswith

    /* String testing operations */
    // isalpha, isdigit, isalnum, islower, isupper, etc.

    /* Split and join operations */
    // split, rsplit, splitlines, partition, rpartition, join

    /* Special operations */
    // translate, encode, decode, compare_chararrays

    /* Array creation operations */
    // array, asarray

} /* namespace ch */
} /* namespace np */
```

### 3. Function Documentation Pattern
```cpp
/**
 * @brief Return element-wise string concatenation for two arrays.
 *
 * Performs element-wise concatenation: result[i] = x1[i] + x2[i].
 * Both input arrays must have identical shapes.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.add.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Element-wise concatenated array
 * @throws std::invalid_argument if x1.shape != x2.shape
 */
NP_API inline auto add(const Ndarray<std::string>& x1,
                        const Ndarray<std::string>& x2)
    -> Ndarray<std::string>
{
    detail::validate_shapes(x1, x2, "char.add");
    
    Ndarray<std::string> result = empty<std::string>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result.data()[i] = x1.data()[i] + x2.data()[i];
    }
    return result;
}
```

### 4. Helper Functions Pattern
```cpp
namespace detail {

/* Validate that two arrays have matching shapes */
inline void validate_shapes(const Ndarray<std::string>& a,
                             const Ndarray<std::string>& b,
                             const char* func_name)
{
    if (a.shape != b.shape) {
        throw std::invalid_argument(
            std::string(func_name) + ": shape mismatch");
    }
}

/* Check if string contains cased characters and all are lowercase */
inline bool str_islower(const std::string& s)
{
    bool has_cased = false;
    for (char c : s) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            has_cased = true;
            if (!std::islower(static_cast<unsigned char>(c))) {
                return false;
            }
        }
    }
    return has_cased;
}

/* Check if string contains cased characters and all are uppercase */
inline bool str_isupper(const std::string& s)
{
    bool has_cased = false;
    for (char c : s) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            has_cased = true;
            if (!std::isupper(static_cast<unsigned char>(c))) {
                return false;
            }
        }
    }
    return has_cased;
}

/* Check if string is titlecased */
inline bool str_istitle(const std::string& s)
{
    bool in_word = false;
    bool has_cased = false;
    
    for (char c : s) {
        bool is_alpha = std::isalpha(static_cast<unsigned char>(c));
        if (is_alpha) {
            has_cased = true;
            if (in_word) {
                if (!std::islower(static_cast<unsigned char>(c))) {
                    return false;
                }
            } else {
                if (!std::isupper(static_cast<unsigned char>(c))) {
                    return false;
                }
                in_word = true;
            }
        } else {
            in_word = false;
        }
    }
    return has_cased;
}

} /* namespace detail */
```

### 5. Comment Style (NO `====`)
```cpp
/* String operations - addition, repetition, formatting */

/* Case conversion operations */

/* Padding and alignment operations */
```

NOT:
```cpp
// ==============================================================
// String Operations
// ==============================================================
```

### 6. Error Messages with Context
```cpp
// BAD:
throw std::invalid_argument("arrays must have the same shape");

// GOOD:
throw std::invalid_argument("char.add: shape mismatch");
```

### 7. Const Correctness
```cpp
// Use const& where appropriate
inline auto lower(const Ndarray<std::string>& a)  // const ref input
    -> Ndarray<std::string>;  // value return (new array)
```

### 8. Consistent Formatting
```cpp
// Brace style
auto func() -> type
{  // Opening brace on new line (project standard)
    // body
}

// Parameter alignment
auto long_function_name(const Ndarray<std::string>& very_long_param1,
                         const Ndarray<std::string>& very_long_param2)
    -> Ndarray<std::string>;

// Indentation: 4 spaces (no tabs)
```

### 9. Macros Usage
```cpp
// Use project macros consistently
NP_API      // Public API functions
NP_INTERNAL // Internal but must be in header
NP_NODISCARD // For pure functions returning new arrays

// Example:
NP_API NP_NODISCARD inline auto upper(const Ndarray<std::string>& a)
    -> Ndarray<std::string>;
```

### 10. Organization by Functionality
Group related functions together with clear comments:

```cpp
/* Case conversion operations */
NP_API inline auto capitalize(...);
NP_API inline auto lower(...);
NP_API inline auto upper(...);
NP_API inline auto swapcase(...);
NP_API inline auto title(...);

/* Padding and alignment operations */
NP_API inline auto center(...);
NP_API inline auto ljust(...);
NP_API inline auto rjust(...);
NP_API inline auto zfill(...);

/* Trimming operations */
NP_API inline auto strip(...);
NP_API inline auto lstrip(...);
NP_API inline auto rstrip(...);
```

## Refactoring Steps

1. ✅ **Backup current file**: `char.hpp` → `char.hpp.backup`

2. **Rewrite file header**:
   - Add proper @file, @brief, @author
   - Document implementation notes
   - Reference numpy docs

3. **Reorganize includes**:
   - Group by category (standard library, project headers)
   - Alphabetize within groups

4. **Create detail namespace**:
   - Move helper functions (str_islower, str_isupper, str_istitle)
   - Add validate_shapes helper
   - Add any other internal utilities

5. **Refactor public functions**:
   - Use detail:: helpers
   - Improve error messages
   - Add NP_NODISCARD where appropriate
   - Ensure const correctness

6. **Update comments**:
   - Replace `====` dividers with `/* ... */` style
   - Ensure every function has proper Doxygen
   - Add @throws documentation

7. **Test refactored version**:
   - Ensure all tests still pass
   - No warnings
   - No behavior changes

## Benefits of Professional Structure

1. **Maintainability**: Clear organization, easy to find functions
2. **Readability**: Consistent style, proper documentation
3. **Extensibility**: Helper functions make it easy to add new operations
4. **Debugging**: Better error messages with context
5. **Professionalism**: Matches industry standards and project conventions
6. **Documentation**: Auto-generates better docs from Doxygen

## Example: Before vs After

### BEFORE (Current)
```cpp
// =================================================================
// String Operations
// =================================================================

/**
 * @brief Return element-wise string concatenation.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.add.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Element-wise concatenation x1[i] + x2[i]
 */
NP_API inline auto add(const Ndarray<std::string>& x1, const Ndarray<std::string>& x2)
    -> Ndarray<std::string> {
    if (x1.shape != x2.shape) {
        throw std::invalid_argument("add: arrays must have the same shape");
    }
    Ndarray<std::string> result = empty<std::string>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result.data()[i] = x1.data()[i] + x2.data()[i];
    }
    return result;
}
```

### AFTER (Professional)
```cpp
/* String operations - addition, repetition, formatting */

/**
 * @brief Return element-wise string concatenation for two arrays.
 *
 * Performs element-wise concatenation: result[i] = x1[i] + x2[i].
 * Both input arrays must have identical shapes.
 *
 * Reference: numpy-reference/reference/generated/numpy.char.add.html
 *
 * @param x1 First string array
 * @param x2 Second string array
 * @return Element-wise concatenated array
 * @throws std::invalid_argument if x1.shape != x2.shape
 */
NP_API inline auto add(const Ndarray<std::string>& x1,
                        const Ndarray<std::string>& x2)
    -> Ndarray<std::string>
{
    detail::validate_shapes(x1, x2, "char.add");
    
    Ndarray<std::string> result = empty<std::string>(x1.shape);
    for (std::size_t i = 0; i < x1.size(); ++i) {
        result.data()[i] = x1.data()[i] + x2.data()[i];
    }
    return result;
}
```

## Implementation Priority

Since the current implementation works (all tests pass), refactoring should be done carefully:

1. **Phase 1** (High Priority): Structure and organization
   - File header
   - Remove `====` dividers
   - Create detail namespace
   - Group functions logically

2. **Phase 2** (Medium Priority): Error handling
   - Better error messages
   - Validate_shapes helper
   - Context in exceptions

3. **Phase 3** (Low Priority): Optimizations
   - NP_NODISCARD where appropriate
   - Const correctness review
   - Performance improvements

## Conclusion

The current implementation is **functionally correct** but needs **stylistic improvements** to match the project's professional standards. The refactoring should focus on:

- **Organization**: Clear structure with detail namespace
- **Documentation**: Professional comments without `====`
- **Consistency**: Match linalg.hpp, math.hpp patterns
- **Maintainability**: Better error messages, helper functions

This can be done incrementally without breaking existing functionality.
