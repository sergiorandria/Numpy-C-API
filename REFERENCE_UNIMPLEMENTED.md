# REFERENCE_UNIMPLEMENTED.md

Gap analysis of the `np::Ndarray` / `np::ndarray` API against the NumPy 2.x
`numpy.ndarray` reference (ground truth: `numpy-reference/reference/generated/`).

Status legend:
- `PLAN` — listed here as a future feature; not yet implemented.
- `DONE` — implemented, tested.

Each row cites its numpy reference page.

## 1. Methods

| numpy.ndarray method | Reference page | Status | C++ API |
|---|---|---|---|
| `all` | `numpy.ndarray.all` | DONE | `all()`, `all(axis)` |
| `any` | `numpy.ndarray.any` | DONE | `any()`, `any(axis)` |
| `argmax` | `numpy.ndarray.argmax` | DONE | `argmax()`, `argmax(axis)` |
| `argmin` | `numpy.ndarray.argmin` | DONE | `argmin()`, `argmin(axis)` |
| `argpartition` | `numpy.ndarray.argpartition` | DONE | `argpartition(kth, axis)` |
| `argsort` | `numpy.ndarray.argsort` | DONE | `argsort(axis)` |
| `astype` | `numpy.ndarray.astype` | DONE | `astype<U>()` |
| `byteswap` | `numpy.ndarray.byteswap` | DONE | `byteswap()` |
| `choose` | `numpy.ndarray.choose` | DONE | `choose(choices, mode)` |
| `clip` | `numpy.ndarray.clip` | DONE | `clip(min, max)` |
| `compress` | `numpy.ndarray.compress` | DONE | `compress(condition, axis)` |
| `conj` | `numpy.ndarray.conj` | DONE | `conj()` |
| `conjugate` | `numpy.ndarray.conjugate` | DONE | `conjugate()` (alias of `conj`) |
| `copy` | `numpy.ndarray.copy` | DONE | `copy()` |
| `cumprod` | `numpy.ndarray.cumprod` | DONE | `cumprod()`, `cumprod(axis)` |
| `cumsum` | `numpy.ndarray.cumsum` | DONE | `cumsum()`, `cumsum(axis)` |
| `diagonal` | `numpy.ndarray.diagonal` | DONE | `diagonal(offset)` |
| `dot` | `numpy.ndarray.dot` | DONE | `dot(b)` (method delegating to `np::dot`) |
| `dump` | `numpy.ndarray.dump` | N/A | pickle of the array to a file |
| `dumps` | `numpy.ndarray.dumps` | N/A | pickle of the array as a string |
| `fill` | `numpy.ndarray.fill` | DONE | `fill(value)` |
| `flatten` | `numpy.ndarray.flatten` | DONE | `flatten()` |
| `getfield` | `numpy.ndarray.getfield` | N/A | typed C++ storage cannot expose byte views of a different dtype |
| `item` | `numpy.ndarray.item` | DONE | `item()` |
| `max` | `numpy.ndarray.max` | DONE | `max()`, `max(axis)` |
| `mean` | `numpy.ndarray.mean` | DONE | `mean()`, `mean(axis)` |
| `min` | `numpy.ndarray.min` | DONE | `min()`, `min(axis)` |
| `nonzero` | `numpy.ndarray.nonzero` | DONE | `nonzero()` |
| `partition` | `numpy.ndarray.partition` | DONE | `partition(kth, axis)` (in-place) |
| `prod` | `numpy.ndarray.prod` | DONE | `prod()`, `prod(axis)` |
| `put` | `numpy.ndarray.put` | DONE | `put(indices, values, mode)` |
| `ravel` | `numpy.ndarray.ravel` | DONE | `ravel()` |
| `repeat` | `numpy.ndarray.repeat` | DONE | `repeat(repeats, axis)` |
| `reshape` | `numpy.ndarray.reshape` | DONE | `reshape(shape)` |
| `resize` | `numpy.ndarray.resize` | DONE | `resize(shape)` |
| `round` | `numpy.ndarray.round` | DONE | `round(decimals)` |
| `searchsorted` | `numpy.ndarray.searchsorted` | DONE | `searchsorted(value)`, `searchsorted(values)` |
| `setfield` | `numpy.ndarray.setfield` | N/A | typed C++ storage cannot write byte fields of a different dtype |
| `setflags` | `numpy.ndarray.setflags` | DONE | `setflags(writeable)` stores a WRITEABLE flag |
| `sort` | `numpy.ndarray.sort` | DONE | `sort(axis)` |
| `squeeze` | `numpy.ndarray.squeeze` | DONE | `squeeze()`, `squeeze(axis)` |
| `std` | `numpy.ndarray.std` | DONE | `std()`, `std(axis)` |
| `sum` | `numpy.ndarray.sum` | DONE | `sum()`, `sum(axis)` |
| `swapaxes` | `numpy.ndarray.swapaxes` | DONE | `swapaxes(a1, a2)` |
| `take` | `numpy.ndarray.take` | DONE | `take(indices, axis)` |
| `tobytes` | `numpy.ndarray.tobytes` | DONE | `tobytes()` |
| `tofile` | `numpy.ndarray.tofile` | DONE | `tofile(filename)`, `tofile(os)` |
| `tolist` | `numpy.ndarray.tolist` | DONE | `tolist()` |
| `trace` | `numpy.ndarray.trace` | DONE | `trace(offset)` |
| `transpose` | `numpy.ndarray.transpose` | DONE | `transpose()`, `transpose(perm)` |
| `var` | `numpy.ndarray.var` | DONE | `var()`, `var(axis)` |
| `view` | `numpy.ndarray.view` | DONE | `view()` |

## 2. Attributes

| numpy.ndarray attribute | Reference page | Status | C++ API |
|---|---|---|---|
| `base` | `numpy.ndarray.base` | DONE | `base()` returns the shared storage owner; `nullptr` when own data |
| `ctypes` | `numpy.ndarray.ctypes` | N/A | Python ctypes integration |
| `data` | `numpy.ndarray.data` | DONE | `data()` |
| `device` | `numpy.ndarray.device` | N/A | array API device (CPU-only library) |
| `dtype` | `numpy.ndarray.dtype` | DONE | `type` member |
| `flags` | `numpy.ndarray.flags` | DONE | `is_contiguous()`, `is_f_contiguous()`, `writeable()`, `owns_data()` |
| `flat` | `numpy.ndarray.flat` | DONE | `flat()` returns a 1-D view (like `ravel`) |
| `imag` | `numpy.ndarray.imag` | DONE | `imag()` — imaginary part (zeros for real types) |
| `itemsize` | `numpy.ndarray.itemsize` | DONE | `itemsize()` |
| `mT` | `numpy.ndarray.mT` | DONE | `mT()` — transpose of the last two dimensions |
| `nbytes` | `numpy.ndarray.nbytes` | DONE | `nbytes()` |
| `ndim` | `numpy.ndarray.ndim` | DONE | `ndim()` |
| `real` | `numpy.ndarray.real` | DONE | `real()` — real part (self for real types) |
| `shape` | `numpy.ndarray.shape` | DONE | `shape` member |
| `size` | `numpy.ndarray.size` | DONE | `size()` |
| `strides` | `numpy.ndarray.strides` | DONE | `strides` member |
| `T` | `numpy.ndarray.T` | DONE | `transpose()` |

## 3. Operators (dunder protocol)

| numpy dunder | Reference page | Status | C++ API |
|---|---|---|---|
| `__abs__` | `numpy.ndarray.__abs__` | DONE | `abs()` member |
| `__add__` | `numpy.ndarray.__add__` | DONE | `operator+` |
| `__and__` | `numpy.ndarray.__and__` | DONE | `operator&` (integral/bool only) |
| `__bool__` | `numpy.ndarray.__bool__` | DONE | `explicit operator bool` |
| `__complex__` | `numpy.ndarray.__complex__` | DONE | `explicit operator std::complex<...>` |
| `__contains__` | `numpy.ndarray.__contains__` | DONE | `contains(value)` |
| `__copy__` | `numpy.ndarray.__copy__` | DONE | `copy()` |
| `__deepcopy__` | `numpy.ndarray.__deepcopy__` | DONE | `copy()` |
| `__divmod__` | `numpy.ndarray.__divmod__` | DONE | `divmod(rhs)` -> `std::pair` (floor_div, mod) |
| `__eq__` | `numpy.ndarray.__eq__` | DONE | `operator==` (elementwise bool) |
| `__float__` | `numpy.ndarray.__float__` | DONE | `explicit operator double` |
| `__floordiv__` | `numpy.ndarray.__floordiv__` | DONE | `floordiv(rhs)` (C++ has no `//`) |
| `__ge__` | `numpy.ndarray.__ge__` | DONE | `operator>=` |
| `__getitem__` | `numpy.ndarray.__getitem__` | DONE | `operator[]`, `operator()`, `get` |
| `__gt__` | `numpy.ndarray.__gt__` | DONE | `operator>` |
| `__iadd__` | `numpy.ndarray.__iadd__` | DONE | `operator+=` |
| `__iand__` | `numpy.ndarray.__iand__` | DONE | `operator&=` |
| `__ifloordiv__` | `numpy.ndarray.__ifloordiv__` | DONE | `floordiv_eq(rhs)` |
| `__ilshift__` | `numpy.ndarray.__ilshift__` | DONE | `operator<<=` |
| `__imod__` | `numpy.ndarray.__imod__` | DONE | `operator%=` |
| `__imul__` | `numpy.ndarray.__imul__` | DONE | `operator*=` |
| `__int__` | `numpy.ndarray.__int__` | DONE | `explicit operator long long` |
| `__invert__` | `numpy.ndarray.__invert__` | DONE | `operator~` (integral/bool only) |
| `__ior__` | `numpy.ndarray.__ior__` | DONE | `operator|=` |
| `__ipow__` | `numpy.ndarray.__ipow__` | DONE | `pow_eq(rhs)` |
| `__irshift__` | `numpy.ndarray.__irshift__` | DONE | `operator>>=` |
| `__isub__` | `numpy.ndarray.__isub__` | DONE | `operator-=` |
| `__itruediv__` | `numpy.ndarray.__itruediv__` | DONE | `operator/=` |
| `__ixor__` | `numpy.ndarray.__ixor__` | DONE | `operator^=` |
| `__le__` | `numpy.ndarray.__le__` | DONE | `operator<=` |
| `__len__` | `numpy.ndarray.__len__` | DONE | `len()` — size of the first dimension |
| `__lshift__` | `numpy.ndarray.__lshift__` | DONE | `operator<<` (shift, integral only) |
| `__lt__` | `numpy.ndarray.__lt__` | DONE | `operator<` |
| `__matmul__` | `numpy.ndarray.__matmul__` | DONE | `matmul(b)` (C++ has no `@`) |
| `__mod__` | `numpy.ndarray.__mod__` | DONE | `operator%` (floored remainder) |
| `__mul__` | `numpy.ndarray.__mul__` | DONE | `operator*` |
| `__ne__` | `numpy.ndarray.__ne__` | DONE | `operator!=` |
| `__neg__` | `numpy.ndarray.__neg__` | DONE | `operator-` (unary) |
| `__or__` | `numpy.ndarray.__or__` | DONE | `operator|` |
| `__pos__` | `numpy.ndarray.__pos__` | DONE | `operator+` (unary) |
| `__pow__` | `numpy.ndarray.__pow__` | DONE | `pow(rhs)` (C++ has no `**`) |
| `__rshift__` | `numpy.ndarray.__rshift__` | DONE | `operator>>` (shift, integral only) |
| `__setitem__` | `numpy.ndarray.__setitem__` | DONE | `set`, `operator[]` |
| `__sub__` | `numpy.ndarray.__sub__` | DONE | `operator-` |
| `__truediv__` | `numpy.ndarray.__truediv__` | DONE | `operator/` |
| `__xor__` | `numpy.ndarray.__xor__` | DONE | `operator^` |

## 4. N/A (Python-only, documented but not implementable here)

- `dump` / `dumps` — pickle protocol.
- `getfield` / `setfield` — byte-offset views into a differently-typed field;
  the library stores one typed `std::vector<T>` per array.
- `ctypes`, `device`, `to_device` — Python C-level / accelerator integration.
- `__class_getitem__`, `__new__`, `__reduce__`, `__setstate__`,
  `__array__`, `__array_wrap__` — Python object protocol.

## 5. Free-function equivalents already provided

`np::choose`, `np::compress`, `np::partition`, `np::setflags` are the numpy
top-level equivalents of the missing ndarray methods above. The library
already ships `np::dot`, `np::matmul`, `np::power`, `np::mod`, `np::fmod`,
`np::floor_divide`-style `np::remainder`, bitwise helpers in `logic.hpp`, and
`np::real` / `np::imag` are planned next to the member methods.

Implementation plan (all items now DONE):

- [x] `choose(choices, mode)` — index-driven selection with broadcasting
- [x] `compress(condition, axis)` — slice selection along an axis
- [x] `conjugate()` — alias of `conj`
- [x] `dot(b)` / `matmul(b)` — delegate to `np::dot` / `np::matmul`
- [x] `partition(kth, axis)` — in-place introselect
- [x] `real()` / `imag()` — complex parts
- [x] `mT()` — last-two-axis transpose
- [x] `setflags(writeable)` + `writeable()` + `is_f_contiguous()` + `owns_data()`
- [x] `flat()` — 1-D view
- [x] `base()` — storage owner pointer
- [x] `abs()`, `len()`, `contains(value)`
- [x] operators `%`, `&`, `|`, `^`, `~`, `<<`, `>>`, unary `+`, in-place variants
- [x] `floordiv`, `divmod`, `pow`, `pow_eq`, `floordiv_eq`
- [x] explicit `bool`, `long long`, `double`, `complex` conversions
