# NumPy C++ API Coverage Status

**Legend**:
- ✅ Fully implemented and tested
- ⚠️ Partially implemented or has limitations
- ❌ Not implemented
- 🔄 Implemented but differs from NumPy (see notes)
- 🚫 Intentionally not implemented (not applicable to C++)

Last Updated: Current Session

---

## Array Creation Functions

### Basic Constructors
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `array()` | ❌ | - | Generic factory needed |
| `asarray()` | ❌ | - | Conversion function needed |
| `asanyarray()` | ❌ | - | |
| `zeros()` | ✅ | creation.hpp | |
| `ones()` | ✅ | creation.hpp | |
| `empty()` | ✅ | creation.hpp | |
| `full()` | ✅ | creation.hpp | |
| `zeros_like()` | ✅ | creation.hpp | |
| `ones_like()` | ✅ | creation.hpp | |
| `empty_like()` | ✅ | creation.hpp | |
| `full_like()` | ✅ | creation.hpp | |

### Ranges
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `arange()` | ✅ | creation.hpp | |
| `linspace()` | ✅ | creation.hpp | |
| `logspace()` | ❌ | - | |
| `geomspace()` | ❌ | - | |

### Matrices
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `eye()` | ✅ | creation.hpp | |
| `identity()` | ✅ | creation.hpp | |
| `diag()` | ✅ | manipulation.hpp | |
| `diagflat()` | ✅ | manipulation.hpp | |
| `tri()` | ✅ | manipulation.hpp | |
| `tril()` | ✅ | manipulation.hpp | |
| `triu()` | ✅ | manipulation.hpp | |
| `vander()` | ✅ | manipulation.hpp | |

### Coordinate Arrays
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `meshgrid()` | ❌ | - | Important for plotting |
| `mgrid` | ❌ | - | |
| `ogrid` | ❌ | - | |
| `indices()` | ❌ | - | |

### From Data
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `frombuffer()` | ❌ | - | Low priority |
| `fromfunction()` | ❌ | - | |
| `fromiter()` | ❌ | - | |
| `fromstring()` | 🚫 | - | Python-specific |

**Coverage**: ~55% (12/22 applicable functions)

---

## Array Manipulation

### Shape Manipulation
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `reshape()` | ✅ | ndarray.hpp | Method |
| `ravel()` | ✅ | ndarray.hpp | Method |
| `flat` | ⚠️ | ndarray.hpp | Iterator exists |
| `flatten()` | ✅ | ndarray.hpp | Method |
| `squeeze()` | ⚠️ | ndarray.hpp | Method, partial |
| `expand_dims()` | ❌ | - | Needed |
| `atleast_1d()` | ❌ | - | Needed |
| `atleast_2d()` | ❌ | - | Needed |
| `atleast_3d()` | ❌ | - | Needed |

### Transpose Operations
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `transpose()` | ✅ | ndarray.hpp | Method and free function |
| `swapaxes()` | ✅ | ndarray.hpp | Method |
| `moveaxis()` | ❌ | - | Needed |
| `rollaxis()` | ❌ | - | |
| `permute_dims()` | ❌ | - | |

### Changing Dimensions
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `broadcast_to()` | ❌ | - | Important |
| `broadcast_arrays()` | ❌ | - | Important |
| `tile()` | ✅ | manipulation.hpp | |
| `repeat()` | ✅ | ndarray.hpp | Method |

### Joining Arrays
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `concatenate()` | ✅ | concatenate.hpp | |
| `stack()` | ⚠️ | concatenate.hpp | Basic version |
| `vstack()` | ✅ | concatenate.hpp | |
| `hstack()` | ✅ | concatenate.hpp | |
| `dstack()` | ✅ | concatenate.hpp | |
| `column_stack()` | ❌ | - | |
| `row_stack()` | ❌ | - | |

### Splitting Arrays
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `split()` | ✅ | manipulation.hpp | |
| `array_split()` | ✅ | manipulation.hpp | |
| `hsplit()` | ✅ | manipulation.hpp | |
| `vsplit()` | ✅ | manipulation.hpp | |
| `dsplit()` | ✅ | manipulation.hpp | |

### Rearranging
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `flip()` | ✅ | manipulation.hpp | |
| `fliplr()` | ✅ | manipulation.hpp | |
| `flipud()` | ✅ | manipulation.hpp | |
| `roll()` | ✅ | manipulation.hpp | |
| `rot90()` | ✅ | manipulation.hpp | |

### Adding/Removing
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `delete()` | ✅ | manipulation.hpp | As delete_arr |
| `insert()` | ⚠️ | manipulation.hpp | Flat only |
| `append()` | ⚠️ | manipulation.hpp | Flat only |
| `resize()` | ✅ | ndarray.hpp | Method |
| `trim_zeros()` | ✅ | manipulation.hpp | |
| `unique()` | ✅ | manipulation.hpp | |
| `pad()` | ❌ | - | **Important!** |

### Selection
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `where()` | ✅ | manipulation.hpp | Both forms |
| `select()` | ❌ | - | |
| `choose()` | ❌ | - | |
| `take()` | ✅ | ndarray.hpp | Method |
| `compress()` | ❌ | - | |
| `extract()` | ❌ | - | |
| `place()` | ❌ | - | |
| `put()` | ✅ | ndarray.hpp | Method |

**Coverage**: ~60% (29/48 functions)

---

## Mathematical Functions

### Arithmetic
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `add()` | ✅ | math.hpp | Operator+ |
| `subtract()` | ✅ | math.hpp | Operator- |
| `multiply()` | ✅ | math.hpp | Operator* |
| `divide()` | ✅ | math.hpp | Operator/ |
| `power()` | ✅ | math.hpp | |
| `mod()` | ✅ | math.hpp | Operator% |
| `negative()` | ✅ | math.hpp | Operator- |
| `positive()` | ✅ | math.hpp | Operator+ |
| `absolute()` | ✅ | math.hpp | |
| `sign()` | ❌ | - | |
| `clip()` | ❌ | - | Useful |
| `sqrt()` | ✅ | math.hpp | |
| `square()` | ✅ | math.hpp | |
| `reciprocal()` | ❌ | - | |

### Trigonometric
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `sin()` | ✅ | math.hpp | |
| `cos()` | ✅ | math.hpp | |
| `tan()` | ✅ | math.hpp | |
| `arcsin()` | ✅ | math.hpp | |
| `arccos()` | ✅ | math.hpp | |
| `arctan()` | ✅ | math.hpp | |
| `arctan2()` | ✅ | math.hpp | |
| `hypot()` | ❌ | - | |
| `sinh()` | ✅ | math.hpp | |
| `cosh()` | ✅ | math.hpp | |
| `tanh()` | ✅ | math.hpp | |
| `arcsinh()` | ✅ | math.hpp | |
| `arccosh()` | ✅ | math.hpp | |
| `arctanh()` | ✅ | math.hpp | |
| `deg2rad()` | ✅ | math.hpp | |
| `rad2deg()` | ✅ | math.hpp | |

### Exponential & Logarithmic
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `exp()` | ✅ | math.hpp | |
| `exp2()` | ✅ | math.hpp | |
| `log()` | ✅ | math.hpp | |
| `log2()` | ✅ | math.hpp | |
| `log10()` | ✅ | math.hpp | |
| `log1p()` | ✅ | math.hpp | |
| `expm1()` | ✅ | math.hpp | |

### Rounding
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `round()` | ✅ | math.hpp | |
| `floor()` | ✅ | math.hpp | |
| `ceil()` | ✅ | math.hpp | |
| `trunc()` | ✅ | math.hpp | |
| `rint()` | ❌ | - | |
| `fix()` | ❌ | - | |

### Sums & Products
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `sum()` | ✅ | ndarray.hpp | Method |
| `prod()` | ✅ | ndarray.hpp | Method |
| `cumsum()` | ❌ | - | Needed |
| `cumprod()` | ❌ | - | Needed |
| `diff()` | ❌ | - | Useful |
| `ediff1d()` | ❌ | - | |
| `gradient()` | ❌ | - | |
| `trapezoid()` | ❌ | - | |

### Extrema
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `maximum()` | ✅ | math.hpp | |
| `minimum()` | ✅ | math.hpp | |
| `fmax()` | ❌ | - | |
| `fmin()` | ❌ | - | |
| `max()` | ✅ | ndarray.hpp | Method |
| `min()` | ✅ | ndarray.hpp | Method |
| `amax()` | ✅ | ndarray.hpp | Alias of max |
| `amin()` | ✅ | ndarray.hpp | Alias of min |
| `argmax()` | ✅ | ndarray.hpp | Method |
| `argmin()` | ✅ | ndarray.hpp | Method |

**Coverage**: ~75% (45/60 common functions)

---

## Linear Algebra

### Matrix Operations
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `dot()` | ✅ | linalg.hpp | |
| `matmul()` | ✅ | linalg.hpp | |
| `inner()` | ✅ | linalg.hpp | |
| `outer()` | ✅ | linalg.hpp | |
| `tensordot()` | ❌ | - | |
| `kron()` | ❌ | - | |
| `cross()` | ❌ | - | |

### Decompositions
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `cholesky()` | ✅ | linalg.hpp | |
| `qr()` | ✅ | linalg.hpp | |
| `svd()` | ✅ | linalg.hpp | |
| `eig()` | ✅ | linalg.hpp | |
| `eigh()` | ⚠️ | linalg.hpp | Symmetric |
| `eigvals()` | ✅ | linalg.hpp | |
| `eigvalsh()` | ⚠️ | linalg.hpp | Symmetric |
| `svdvals()` | ✅ | linalg.hpp | |

### Matrix Properties
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `det()` | ✅ | linalg.hpp | |
| `slogdet()` | ❌ | - | |
| `matrix_rank()` | ✅ | linalg.hpp | |
| `norm()` | ✅ | linalg.hpp | Multiple norms |
| `cond()` | ✅ | linalg.hpp | |
| `trace()` | ✅ | linalg.hpp | |

### Solving
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `solve()` | ✅ | linalg.hpp | |
| `lstsq()` | ✅ | linalg.hpp | |
| `inv()` | ✅ | linalg.hpp | |
| `pinv()` | ✅ | linalg.hpp | |

**Coverage**: ~80% (20/25 functions)

---

## Logic & Comparison

### Comparison
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `equal()` | ✅ | logic.hpp | Operator== |
| `not_equal()` | ✅ | logic.hpp | Operator!= |
| `greater()` | ✅ | logic.hpp | Operator> |
| `greater_equal()` | ✅ | logic.hpp | Operator>= |
| `less()` | ✅ | logic.hpp | Operator< |
| `less_equal()` | ✅ | logic.hpp | Operator<= |
| `isclose()` | ❌ | - | Important |
| `allclose()` | ❌ | - | Important |
| `array_equal()` | ❌ | - | |
| `array_equiv()` | ❌ | - | |

### Logical Operations
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `logical_and()` | ✅ | logic.hpp | |
| `logical_or()` | ✅ | logic.hpp | |
| `logical_not()` | ✅ | logic.hpp | |
| `logical_xor()` | ✅ | logic.hpp | |

### Truth Testing
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `all()` | ✅ | logic.hpp | |
| `any()` | ✅ | logic.hpp | |
| `isfinite()` | ❌ | - | |
| `isinf()` | ❌ | - | |
| `isnan()` | ❌ | - | |

**Coverage**: ~65% (13/20 functions)

---

## Random Sampling

### Distributions
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `random()` | ✅ | random.hpp | |
| `randn()` | ✅ | random.hpp | |
| `randint()` | ✅ | random.hpp | |
| `uniform()` | ✅ | random.hpp | |
| `normal()` | ✅ | random.hpp | |
| `binomial()` | ✅ | random.hpp | |
| `poisson()` | ✅ | random.hpp | |
| `exponential()` | ✅ | random.hpp | |
| `gamma()` | ✅ | random.hpp | |
| `beta()` | ✅ | random.hpp | |
| `chi_square()` | ✅ | random.hpp | |
| `f()` | ✅ | random.hpp | |
| `geometric()` | ✅ | random.hpp | |
| `weibull()` | ✅ | random.hpp | |
| ...and 15+ more | ✅ | random.hpp | See random.hpp |

**Coverage**: ~95% (30+ distributions implemented)

---

## String Operations (numpy.char)

### String Operations
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `add()` | ✅ | char.hpp | Concatenation |
| `multiply()` | ✅ | char.hpp | String repetition |
| `mod()` | ✅ | char.hpp | String formatting |
| `capitalize()` | ✅ | char.hpp | |
| `center()` | ✅ | char.hpp | |
| `lower()` | ✅ | char.hpp | |
| `upper()` | ✅ | char.hpp | |
| `strip()` | ✅ | char.hpp | |
| `lstrip()` | ✅ | char.hpp | |
| `rstrip()` | ✅ | char.hpp | |
| `swapcase()` | ✅ | char.hpp | |
| `title()` | ✅ | char.hpp | |
| `zfill()` | ✅ | char.hpp | |
| `ljust()` | ✅ | char.hpp | |
| `rjust()` | ✅ | char.hpp | |
| `replace()` | ✅ | char.hpp | |
| `expandtabs()` | ✅ | char.hpp | |
| `partition()` | ✅ | char.hpp | |
| `rpartition()` | ✅ | char.hpp | |
| `split()` | ✅ | char.hpp | Flattened result |
| `rsplit()` | ✅ | char.hpp | Flattened result |
| `splitlines()` | ✅ | char.hpp | Flattened result |
| `translate()` | ✅ | char.hpp | Simplified |
| `join()` | ✅ | char.hpp | |
| `encode()` | ✅ | char.hpp | No-op in C++ |
| `decode()` | ✅ | char.hpp | No-op in C++ |

### Comparison
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `equal()` | ✅ | char.hpp | |
| `not_equal()` | ✅ | char.hpp | |
| `greater_equal()` | ✅ | char.hpp | |
| `less_equal()` | ✅ | char.hpp | |
| `greater()` | ✅ | char.hpp | |
| `less()` | ✅ | char.hpp | |
| `compare_chararrays()` | ✅ | char.hpp | |

### Information
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `count()` | ✅ | char.hpp | |
| `endswith()` | ✅ | char.hpp | |
| `startswith()` | ✅ | char.hpp | |
| `find()` | ✅ | char.hpp | |
| `rfind()` | ✅ | char.hpp | |
| `index()` | ✅ | char.hpp | |
| `rindex()` | ✅ | char.hpp | |
| `str_len()` | ✅ | char.hpp | |

### Testing
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `isalpha()` | ✅ | char.hpp | |
| `isalnum()` | ✅ | char.hpp | |
| `isdecimal()` | ✅ | char.hpp | Uses isdigit |
| `isdigit()` | ✅ | char.hpp | |
| `islower()` | ✅ | char.hpp | |
| `isnumeric()` | ✅ | char.hpp | Uses isdigit |
| `isspace()` | ✅ | char.hpp | |
| `istitle()` | ✅ | char.hpp | |
| `isupper()` | ✅ | char.hpp | |

### Creation
| Function | Status | File | Notes |
|----------|--------|------|-------|
| `array()` | ✅ | char.hpp | |
| `asarray()` | ✅ | char.hpp | |
| `chararray` | 🚫 | - | Deprecated in NumPy 2.5 |

**Coverage**: 100% (53/53 functions)

---

## FFT

| Function | Status | File | Notes |
|----------|--------|------|-------|
| `fft()` | ✅ | fft.hpp | |
| `ifft()` | ✅ | fft.hpp | |
| `fft2()` | ✅ | fft.hpp | |
| `ifft2()` | ✅ | fft.hpp | |
| `fftn()` | ✅ | fft.hpp | |
| `ifftn()` | ✅ | fft.hpp | |
| `rfft()` | ✅ | fft.hpp | |
| `irfft()` | ✅ | fft.hpp | |
| `fftshift()` | ✅ | fft.hpp | |
| `ifftshift()` | ✅ | fft.hpp | |

**Coverage**: ~90% (10/11 main FFT functions)

---

## Overall Summary

| Category | Implemented | Total | Percentage |
|----------|-------------|-------|------------|
| Array Creation | 12 | 22 | 55% |
| Array Manipulation | 29 | 48 | 60% |
| Mathematical | 45 | 60 | 75% |
| Linear Algebra | 20 | 25 | 80% |
| Logic & Comparison | 13 | 20 | 65% |
| Random | 30 | 32 | 95% |
| String Operations (char) | 53 | 53 | 100% |
| FFT | 10 | 11 | 90% |
| **TOTAL** | **212** | **271** | **78%** |

---

## High Priority Missing Functions

1. `pad()` - Array padding
2. `clip()` - Value clipping
3. `isclose()` / `allclose()` - Floating point comparison
4. `cumsum()` / `cumprod()` - Cumulative operations
5. `diff()` - Differences
6. `meshgrid()` - Coordinate arrays
7. `expand_dims()` - Dimension expansion
8. `atleast_Nd()` - Dimension guarantees
9. `broadcast_to()` - Broadcasting
10. `moveaxis()` - Axis manipulation

---

## Intentionally Not Implemented

- String/character arrays (`numpy.char.*`) - Not core numerical
- Masked arrays (`numpy.ma.*`) - Complex feature, defer
- Polynomial functions - Low priority
- I/O functions (`load`, `save`, `fromfile`) - Platform dependent
- Datetime functions - Not numerical computing
- Financial functions - Specialized
