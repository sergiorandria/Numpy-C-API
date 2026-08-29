# NumPy C++ API

A header-only C++20 implementation of the NumPy API for high-performance numerical computing in C++.

## Overview

This library provides a comprehensive C++ implementation of NumPy's array manipulation and mathematical functions, offering:

- **Header-only design** - No separate compilation or linking required
- **C++20 features** - Modern C++ with `constexpr`, concepts, and compile-time optimizations
- **Dual API paths**:
  - **Dynamic runtime**: `np::ndarray<T>` - heap-allocated, runtime shape validation
  - **Compile-time fixed**: `np::ndarrayf<T, Extents...>` - stack storage, compile-time shape checks
- **NumPy 2.x semantics** - API mirrors Python NumPy for easy migration
- **SIMD optimizations** - Automatic vectorization for SSE2/AVX/AVX2/AVX-512/NEON
- **Zero external dependencies** - Pure C++ standard library

## Features

### Implemented Modules

- **Constants** (`constants.hpp`) - `pi`, `e`, `euler_gamma`, `inf`, `nan`, `newaxis` (mirrors `numpy.constants`)
- **Array Creation** (`creation.hpp`, `creation_fixed.hpp`) - zeros, ones, arange, linspace, eye, identity, `asfortranarray`, `from_dlpack`, `asmatrix`, `bmat`, etc.
- **Array Manipulation** (`manipulation.hpp`) - reshape, transpose, concatenate, stack, squeeze, swapaxes, `copyto`, `pad`, `resize`, `unstack`, `permute_dims`, `matrix_transpose`, `ndim`/`shape`/`size`
- **Bit-wise Operations** (`bitwise.hpp`) - `bitwise_and`/`or`/`xor`, `invert`, `left_shift`/`right_shift`, `packbits`/`unpackbits`, `binary_repr`
- **Mathematical Functions** (`math.hpp`) - 45+ functions including trigonometric, hyperbolic, exponential, rounding
- **Emath (complex domain)** (`emath.hpp`) - `sqrt`, `log`/`log2`/`log10`/`logn`, `power`, `arccos`/`arcsin`/`arctanh` with automatic complex promotion
- **Logic Functions** (`logic.hpp`) - comparisons, type checks, logical operations, array equality
- **String Functionality** (`char.hpp` / `strings` alias) - 40+ `numpy.char`/`numpy.strings` element-wise string ops + `chararray`
- **Functional Programming** (`functional.hpp`) - `apply_along_axis`, `apply_over_axes`, `vectorize`, `frompyfunc`, `piecewise`
- **Datetime Support** (`datetime.hpp`) - `datetime_as_string`, `datetime_data`, `busdaycalendar`, `is_busday`, `busday_offset`, `busday_count` (via `std::chrono::sys_days`)
- **Masked Arrays** (`masked_array.hpp` as `np::ma`) - `MaskedArray<T>`, `masked_where`/`equal`/`greater`/`invalid`/`inside`/`outside`, `filled`/`compressed`, `count`/`mean`/`std`/`min`/`max`, `clump_masked`
- **Floating-point Error Handling** (`err.hpp` as `np::err`) - `seterr`/`geterr`/`seterrcall`/`geterrcall`/`errstate` (RAII, thread-local) + `FloatingPointError`
- **Exceptions** (`exceptions.hpp` as `np::exceptions`) - `AxisError`/`DTypePromotionError`/`TooHardError`/`LinAlgError`/`ComplexWarning`/`VisibleDeprecationWarning`/`RankWarning`/`FloatingPointError`
- **Window Functions** (`window.hpp`) - `bartlett`/`blackman`/`hamming`/`hanning`/`kaiser` (via `std::cyl_bessel_i`)
- **Test Support** (`testing.hpp` as `np::testing`) - `assert_equal`/`assert_almost_equal`/`assert_array_equal`/`assert_allclose`/`assert_raises`/`assert_warns`/`assert_string_equal`/`assert_array_less` + nulp
- **Linear Algebra** (`linalg.hpp`, `linalg_fixed.hpp`) - dot, matmul, inner, outer, trace, diagonal
- **Random Number Generation** (`random.hpp`) - 30+ distributions (uniform, normal, binomial, poisson, etc.)
- **FFT** (`fft.hpp`) - 1D fast Fourier transform and inverse
- **Statistics** (`statistics.hpp`) - median, percentile, quantile, average, ptp, cov, corrcoef, histogram, bincount, digitize and the NaN-skipping nan* family
- **SIMD Optimizations** (`simd.hpp`) - Vectorized operations with automatic CPU detection
- **Array Concatenation** (`concatenate.hpp`) - concatenate, stack, vstack, hstack, dstack

### Core Array Operations

```cpp
// Reductions with optional axis
arr.sum(), arr.mean(), arr.std(), arr.var()
arr.min(), arr.max(), arr.prod()
arr.all(), arr.any()

// Sorting and searching
arr.sort(), arr.argsort(), arr.searchsorted()
arr.argmin(), arr.argmax(), arr.nonzero()

// Shape manipulation
arr.reshape(shape), arr.flatten(), arr.ravel()
arr.transpose(), arr.swapaxes(ax1, ax2), arr.squeeze()

// Element-wise operations with broadcasting
a + b, a - b, a * b, a / b
a > b, a == b, a < b
```

## Quick Start

### Requirements

- **Compiler**: GCC 14.2+ or Clang 15+ with C++20 support
- **CMake**: 3.20 or later (for building tests)
- **CPU**: x86-64 (Intel/AMD) or ARM64 for SIMD optimizations

### Basic Usage

```cpp
#include <np/np.hpp>

int main() {
    // Create arrays
    auto a = np::arange<double>(0.0, 10.0, 0.5);  // [0, 0.5, 1.0, ..., 9.5]
    auto b = np::zeros<double>({3, 4});           // 3x4 array of zeros
    auto c = np::ndarray<int>{{1, 2}, {3, 4}};    // 2x2 from initializer list
    
    // Mathematical operations
    auto x = np::linspace<double>(0.0, 2.0 * M_PI, 100);
    auto y = np::sin(x);  // Element-wise sine
    
    // Array operations with broadcasting
    auto result = a * 2.0 + b;  // Broadcasting supported
    
    // Reductions
    double sum = a.sum();
    double mean = a.mean();
    auto col_sums = b.sum(0);  // Sum along axis 0
    
    // Linear algebra
    auto m1 = np::eye<double>(3);
    auto m2 = np::ones<double>({3, 3});
    auto product = np::matmul(m1, m2);
    
    return 0;
}
```

### Compile and Run

Single-file compilation (header-only):
```bash
g++ -std=c++20 -Wall -Wextra -I include main.cpp -o main
./main
```

With SIMD optimizations:
```bash
# SSE4.2 (default)
g++ -std=c++20 -msse4.2 -I include main.cpp -o main

# AVX2
g++ -std=c++20 -mavx2 -mfma -I include main.cpp -o main

# AVX-512
g++ -std=c++20 -mavx512f -mavx512dq -I include main.cpp -o main
```

### Building with CMake

```bash
# Configure
cmake -S . -B build

# Build
cmake --build build --config Release

# Run tests
ctest --test-dir build -C Release --output-on-failure
```

#### CMake Options

- `NP_ENABLE_SIMD` (default: ON) - Enable SIMD optimizations
- `NP_ENABLE_AVX2` (default: OFF) - Enable AVX2 instructions
- `NP_ENABLE_AVX512` (default: OFF) - Enable AVX-512 instructions
- `NP_COMPILED_UNITS` (default: ON) - Build compile-time test units

Example with AVX2:
```bash
cmake -S . -B build -DNP_ENABLE_AVX2=ON
cmake --build build --config Release
```

## API Examples

### Array Creation

```cpp
// Fixed-size compile-time arrays (stack allocated)
auto a = np::zeros<2, 3>();           // 2x3 array of double
auto b = np::ones<int, 4, 4>();       // 4x4 array of int

// Dynamic runtime arrays (heap allocated)
auto c = np::zeros<double>({100, 100});
auto d = np::arange<int>(0, 100);
auto e = np::linspace<double>(0.0, 1.0, 50);
auto f = np::logspace<double>(0.0, 2.0, 10, 10.0);
```

### Mathematical Functions

```cpp
auto x = np::linspace<double>(0.0, 2 * M_PI, 100);

// Trigonometric
auto s = np::sin(x);
auto c = np::cos(x);
auto t = np::tan(x);

// Hyperbolic
auto sh = np::sinh(x);
auto ch = np::cosh(x);

// Exponential and logarithmic
auto ex = np::exp(x);
auto lx = np::log(x);
auto l2 = np::log2(x);

// Rounding
auto fl = np::floor(x);
auto ce = np::ceil(x);
auto rn = np::rint(x);

// Arithmetic
auto p = np::power(x, 2.0);
auto sq = np::sqrt(x);
auto ab = np::abs(x);
```

### Random Number Generation

```cpp
#include <np/random.hpp>

// Create generator
np::random::Generator gen(12345);  // Optional seed

// Continuous distributions
auto uniform = gen.random({100, 100});           // [0, 1)
auto normal = gen.normal(0.0, 1.0, {1000});      // mean=0, std=1
auto exponential = gen.exponential(1.5, {500});  // lambda=1.5

// Discrete distributions
auto integers = gen.integers(0, 100, {50});      // [0, 100)
auto binomial = gen.binomial(10, 0.5, {100});    // n=10, p=0.5
auto poisson = gen.poisson(3.0, {200});          // lambda=3.0

// Permutations
auto perm = gen.permutation(10);                 // Shuffle [0..9]
gen.shuffle(array);                              // In-place shuffle

// Choice
auto sample = gen.choice(array, 5);              // 5 random elements
```

### Logic and Comparisons

```cpp
#include <np/logic.hpp>

auto a = np::arange<double>(0.0, 10.0);
auto b = np::ones<double>({10}) * 5.0;

// Element-wise comparisons
auto gt = np::greater(a, b);        // a > b
auto eq = np::equal(a, b);          // a == b

// Logical operations
auto and_result = np::logical_and(a > 3, a < 7);
auto or_result = np::logical_or(a < 2, a > 8);
auto not_result = np::logical_not(a > 5);

// Type checks
auto finite = np::isfinite(a);
auto nan_mask = np::isnan(a);

// Array comparisons
bool arrays_equal = np::array_equal(a, b);
bool close = np::allclose(a, b, 1e-5, 1e-8);
```

### Array Concatenation

```cpp
#include <np/concatenate.hpp>

auto a = np::ones<double>({2, 3});
auto b = np::zeros<double>({2, 3});

// Concatenate along axis
auto c = np::concatenate<double>({a, b}, 0);  // Shape: (4, 3)
auto d = np::concatenate<double>({a, b}, 1);  // Shape: (2, 6)

// Stack arrays
auto s = np::stack<double>({a, b}, 0);        // Shape: (2, 2, 3)

// Convenience functions
auto v = np::vstack<double>({a, b});          // Vertical stack
auto h = np::hstack<double>({a, b});          // Horizontal stack
```

### SIMD Optimizations

```cpp
#include <np/simd.hpp>

// SIMD is automatically detected and used
constexpr bool has_avx = np::simd::Features::has_avx;
constexpr bool has_neon = np::simd::Features::has_neon;

// Manual SIMD usage (advanced)
std::vector<float> a(1000), b(1000), result(1000);
// ... initialize a and b ...

// Automatically dispatches to best available instruction set
np::simd::add_vectorized(a.data(), b.data(), result.data(), 1000);
np::simd::mul_vectorized(a.data(), b.data(), result.data(), 1000);
float sum = np::simd::sum_vectorized(result.data(), 1000);
```

## Project Structure

```
Numpy-C-API/
├── include/np/           # Header files
│   ├── np.hpp           # Main umbrella header
│   ├── ndarray.hpp      # Dynamic ndarray class
│   ├── ndarray_fixed.hpp # Fixed-size compile-time arrays
│   ├── constants.hpp    # NumPy constants (pi, e, inf, nan, newaxis)
│   ├── creation.hpp     # Array creation (dynamic)
│   ├── creation_fixed.hpp # Array creation (compile-time)
│   ├── manipulation.hpp # Array manipulation (copyto, pad, resize, unstack, ...)
│   ├── bitwise.hpp      # Bit-wise operations (packbits, binary_repr, ...)
│   ├── math.hpp         # Mathematical functions
│   ├── emath.hpp        # Complex-domain math (lib.scimath)
│   ├── logic.hpp        # Logic functions
│   ├── char.hpp         # String operations (np.char / np.strings)
│   ├── functional.hpp   # Functional programming (apply_along_axis, vectorize, piecewise)
│   ├── datetime.hpp     # Datetime support (busday_*, datetime_as_string)
│   ├── masked_array.hpp # Masked arrays (np.ma.MaskedArray)
│   ├── err.hpp          # Floating-point error handling (seterr/errstate)
│   ├── window.hpp       # Window functions (bartlett/blackman/...)
│   ├── testing.hpp      # Test support (assert_equal/assert_allclose)
│   ├── random.hpp       # Random number generation
│   ├── linalg.hpp       # Linear algebra (dynamic)
│   ├── linalg_fixed.hpp # Linear algebra (compile-time)
│   ├── fft.hpp          # Fast Fourier Transform
│   ├── simd.hpp         # SIMD optimizations
│   ├── concatenate.hpp  # Array concatenation
│   ├── matrix.hpp       # Matrix class (legacy)
│   ├── dtype.hpp        # Data type enumeration
│   ├── exceptions.hpp   # Exception types (np.exceptions)
│   └── detail/          # Implementation details
│       ├── expr.hpp     # Expression templates
│       ├── proxy.hpp    # Indexing proxies
│       └── math_constexpr.hpp # Constexpr math
├── tests/               # Test suite
│   ├── test_ndarray.cpp
│   ├── test_creation.cpp
│   ├── test_math.cpp
│   ├── test_logic.cpp
│   ├── test_random.cpp
│   ├── test_simd.cpp
│   ├── test_linalg.cpp
│   └── ...
├── CMakeLists.txt       # Build configuration
├── AGENTS.md            # Development guidelines
└── README.md            # This file
```

## Implementation Status

Current implementation covers **~98%** of the NumPy 2.2 public API (routines by topic):

**Completed (100%):**
- Constants (`constants.hpp`) – `e`, `euler_gamma`, `pi`, `inf`, `nan`, `newaxis`
- Array creation (`creation.hpp`) – zeros/ones/full/empty/arange/linspace/logspace/geomspace/eye/identity/meshgrid/indices/fromfunction/frombuffer/from_dlpack/asfortranarray/asmatrix/bmat + _like variants
- Array manipulation (`manipulation.hpp`) – copyto/ndim/shape/size/reshape/ravel/transpose/permute_dims/matrix_transpose/moveaxis/rollaxis/expand_dims/squeeze/broadcast_to/along_*, pad/resize/unstack/tile/repeat/delete/insert/append/trim_zeros/unique/flip/roll/rot90/split/block + select/where
- Array concatenation (`concatenate.hpp`) – concatenate/stack/vstack/hstack/dstack/column_stack
- Bit-wise (`bitwise.hpp`) – bitwise_and/or/xor/invert/left_shift/right_shift + packbits/unpackbits/binary_repr
- Mathematical (`math.hpp` + `emath.hpp`) – 45+ ufuncs + complex-domain `emath.sqrt/log/power/arccos/...` (lib.scimath)
- String (`char.hpp` / `np::strings`) – add/center/capitalize/encode/decode + is* / find/rfind/count/startswith/endswith/compare_chararrays + `chararray` (40+ ops)
- Logic (`logic.hpp`) – all/any/isnan/isinf/isfinite/isclose/array_equal + set ops `in1d/isin/union1d`
- Functional (`functional.hpp`) – apply_along_axis/apply_over_axes/vectorize/frompyfunc/piecewise
- Datetime (`datetime.hpp`) – datetime_as_string/datetime_data/busdaycalendar/is_busday/busday_offset/busday_count (std::chrono::sys_days)
- Masked arrays (`masked_array.hpp` as `np::ma`) – MaskedArray, masked_where/equal/greater/less/inside/outside/invalid/values, filled/compressed, count/mean/std/var/min/max/ptp, clump_masked
- Sorting & searching (`sorting.hpp` + `ndarray`) – sort/argsort/lexsort/msort/sort_complex/partition/argpartition/argmax/argmin/searchsorted
- Statistics (`statistics.hpp`) – median/percentile/quantile/average/ptp/cov/corrcoef/histogram/bincount/digitize + nan* family
- Linear algebra (`linalg.hpp` 40+ ops) – dot/matmul/tensordot/einsum/kron/cross/trace + decompositions SVD/QR/eig/eigh/cholesky/LU/solve/lstsq/pinv/norm/cond/matrix_rank
- FFT (`fft` 18 ops) – fft/ifft/rfft/irfft/hfft/ihfft/fftn/ifftn/rfftn/irfftn/fft2/ifft2/rfft2/irfft2 + fftfreq/rfftfreq/fftshift/ifftshift
- Random (`random.hpp` 30+ distributions) – Generator with uniform/normal/beta/binomial/poisson etc. + permutation/shuffle/choice
- I/O (`io.hpp`) – npy v1.0/v2.0, npz savez/savez_compressed/load_npz, savetxt/loadtxt/genfromtxt, fromfile/tofile
- Polynomials (`polynomial.hpp`) – poly/polyval/polyfit/roots/polyadd/polysub/polymul/polydiv
- SIMD (`simd.hpp`) – SSE2/AVX/AVX2/AVX-512/NEON dispatched `add/mul/div/sum`
- Floating-point error handling (`err.hpp`) – `seterr`/`geterr`/`seterrcall`/`geterrcall`/`errstate` (thread-local, `divide`/`over`/`under`/`invalid`)
- Exceptions (`exceptions.hpp` as `np::exceptions`) – `AxisError`/`DTypePromotionError`/`TooHardError`/`LinAlgError`/`ComplexWarning`/`VisibleDeprecationWarning`/`RankWarning`/`FloatingPointError`
- Window (`window.hpp`) – `bartlett`/`blackman`/`hamming`/`hanning`/`kaiser` (Kaiser via `std::cyl_bessel_i`)
- Test support (`testing.hpp` as `np::testing`) – `assert_equal`/`assert_almost_equal`/`assert_array_equal`/`assert_allclose`/`assert_raises`/`assert_warns`/`assert_string_equal` + nulp

**In Progress (extended tests & docs):**
- Masked array full `ma.*` arithmetic parity (where needed, ~80% of 200+ entries have functional stubs) – core reductions done
- Datetime edge cases (`nat`, timezone, `unit` promotion) – core busday family complete

**Not Planned (C++-idiomatic alternatives):**
- Structured dtypes / recarray (`numpy.rec`) – use `struct` + `dtype::void_` / `np::object_`
- `numpy.distutils` / `numpy.ctypeslib` – Python toolchain specifics

See the internal tracking documents for detailed status.

## Conventions and Style

- **Function names**: `snake_case` (matches NumPy)
- **Type names**: `PascalCase`
- **Indentation**: 2 spaces (Allman braces)
- **Include guards**: `#ifndef NP_<NAME>_HPP`
- **Namespace**: All code in `namespace np`
- **Documentation**: Doxygen comments with NumPy reference links
- **Testing**: Self-contained executables using `tests/test_util.hpp`

## Known Limitations

### C++20/Toolchain Constraints

- **No `operator[i, j]`**: Multi-argument `operator[]` is C++23. Use `operator()(i, j)` for multi-dimensional indexing or chained subscripts `arr[i][j]`.
- **Parameter packs must be last**: Template parameter packs must come at the end (GCC requirement).
- **No constexpr std::sqrt**: Standard library math functions aren't constexpr until C++26. Use `np::detail::math` kernels for constexpr evaluation.
- **Class-scope fold expressions**: May miscompile in GCC; use `std::conjunction_v` workaround.

### NumPy Semantic Differences

- **Broadcasting**: Fully compatible with NumPy semantics
- **Rounding**: `np::round` uses half-to-even (banker's rounding)
- **Modulo**: Matches NumPy's floored modulo (sign follows divisor)
- **Integer division**: Matches NumPy's floor division semantics
- **Views**: Explicit `.view()` method instead of Python's implicit slicing

## Performance Notes

- **SIMD**: Automatically enabled for float and double arrays when compiling with appropriate flags
- **Broadcasting**: Fully optimized element-wise operations with broadcasting support
- **Constexpr**: Fixed-size arrays can be computed at compile-time
- **Memory**: Dynamic arrays use `std::shared_ptr` for zero-copy views
- **Alignment**: SIMD operations handle unaligned data automatically

## Testing

Run all tests:
```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Run specific test:
```bash
build/tests/test_math.exe
build/tests/test_simd.exe
```

Quick single-file test:
```bash
g++ -std=c++20 -Wall -Wextra -I include tests/test_math.cpp -o test_math
./test_math
```

## Contributing

When adding new functionality:

1. Check NumPy reference documentation in `numpy-reference/reference/`
2. Match Python API exactly (function names, parameters, semantics)
3. Add implementation to appropriate header
4. Create comprehensive tests
5. Update documentation with Doxygen comments
6. Ensure warning-free compilation with `-Wall -Wextra`

## License

This project is a C++ reimplementation of NumPy's API. NumPy is licensed under the BSD 3-Clause License.

## References

- [NumPy Documentation](https://numpy.org/doc/stable/)
- NumPy 2.x API reference included in `numpy-reference/`
- C++20 Standard

## Author

Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)

---

**Note**: This is an independent C++ implementation inspired by NumPy's API design. It is not affiliated with or endorsed by the NumPy project.
