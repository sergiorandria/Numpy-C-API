# NumPy char Module Quick Reference

Quick lookup for `np::ch` (numpy.char) string operations.

## Include & Namespace

```cpp
#include "np/char.hpp"

using namespace np;
using namespace np::ch;  // char module functions
```

## Creation

```cpp
auto arr = array(std::vector<std::string>{"hello", "world"});
auto arr2 = asarray(existing_array);
```

## Case Conversion

| Function | Example | Result |
|----------|---------|--------|
| `lower()` | `lower(["HELLO"])` | `["hello"]` |
| `upper()` | `upper(["hello"])` | `["HELLO"]` |
| `capitalize()` | `capitalize(["hello world"])` | `["Hello world"]` |
| `title()` | `title(["hello world"])` | `["Hello World"]` |
| `swapcase()` | `swapcase(["Hello"])` | `["hELLO"]` |

## Padding & Alignment

| Function | Example | Result |
|----------|---------|--------|
| `center(arr, w, c)` | `center(["ab"], 5, '*')` | `["**ab*"]` |
| `ljust(arr, w, c)` | `ljust(["ab"], 5, '-')` | `["ab---"]` |
| `rjust(arr, w, c)` | `rjust(["ab"], 5, '-')` | `["---ab"]` |
| `zfill(arr, w)` | `zfill(["42"], 5)` | `["00042"]` |

## Trimming

| Function | Example | Result |
|----------|---------|--------|
| `strip()` | `strip(["  hi  "])` | `["hi"]` |
| `lstrip()` | `lstrip(["  hi"])` | `["hi"]` |
| `rstrip()` | `rstrip(["hi  "])` | `["hi"]` |

## String Operations

| Function | Example | Result |
|----------|---------|--------|
| `add(a, b)` | `add(["hello"], [" world"])` | `["hello world"]` |
| `multiply(a, n)` | `multiply(["ab"], [3])` | `["ababab"]` |
| `replace(a, old, new)` | `replace(["hello"], "l", "L")` | `["heLLo"]` |
| `replace(a, old, new, cnt)` | `replace(["hello"], "l", "L", 1)` | `["heLlo"]` |

## Search & Find

| Function | Returns | Example |
|----------|---------|---------|
| `find(arr, sub)` | int array (pos or -1) | `find(["hello"], "l")` → `[2]` |
| `rfind(arr, sub)` | int array (last pos) | `rfind(["hello"], "l")` → `[3]` |
| `index(arr, sub)` | int array (throws if not found) | `index(["hello"], "l")` → `[2]` |
| `rindex(arr, sub)` | int array (throws if not found) | `rindex(["hello"], "l")` → `[3]` |
| `count(arr, sub)` | int array | `count(["hello"], "l")` → `[2]` |
| `str_len(arr)` | int array | `str_len(["hello"])` → `[5]` |

## Checks (Boolean Arrays)

| Function | Checks | Example |
|----------|--------|---------|
| `startswith(arr, prefix)` | Starts with | `startswith(["hello"], "he")` → `[true]` |
| `endswith(arr, suffix)` | Ends with | `endswith(["hello"], "lo")` → `[true]` |
| `isalpha()` | All alphabetic | `isalpha(["abc"])` → `[true]` |
| `isdigit()` | All digits | `isdigit(["123"])` → `[true]` |
| `isalnum()` | All alphanumeric | `isalnum(["abc123"])` → `[true]` |
| `islower()` | All lowercase | `islower(["abc"])` → `[true]` |
| `isupper()` | All uppercase | `isupper(["ABC"])` → `[true]` |
| `isspace()` | All whitespace | `isspace(["   "])` → `[true]` |
| `istitle()` | Titlecased | `istitle(["Hello World"])` → `[true]` |
| `isdecimal()` | All decimal | `isdecimal(["123"])` → `[true]` |
| `isnumeric()` | All numeric | `isnumeric(["123"])` → `[true]` |

## Comparison (Boolean Arrays)

| Function | Operation | Example |
|----------|-----------|---------|
| `equal(a, b)` | a == b | `equal(["a"], ["a"])` → `[true]` |
| `not_equal(a, b)` | a != b | `not_equal(["a"], ["b"])` → `[true]` |
| `less(a, b)` | a < b | `less(["a"], ["b"])` → `[true]` |
| `greater(a, b)` | a > b | `greater(["b"], ["a"])` → `[true]` |
| `less_equal(a, b)` | a <= b | `less_equal(["a"], ["a"])` → `[true]` |
| `greater_equal(a, b)` | a >= b | `greater_equal(["b"], ["a"])` → `[true]` |

## Split & Join

| Function | Description | Example |
|----------|-------------|---------|
| `split(arr)` | Split on whitespace | `split(["a b"])` → `["a", "b"]` |
| `split(arr, sep)` | Split on separator | `split(["a,b"], ",")` → `["a", "b"]` |
| `rsplit(arr, sep)` | Split from right | `rsplit(["a,b,c"], ",")` → `["a", "b", "c"]` |
| `splitlines(arr)` | Split on newlines | `splitlines(["a\nb"])` → `["a", "b"]` |
| `partition(arr, sep)` | 3-way split | `partition(["a-b"], "-")` → `["a", "-", "b"]` |
| `rpartition(arr, sep)` | 3-way from right | `rpartition(["a-b"], "-")` → `["a", "-", "b"]` |
| `join(sep, seq)` | Join with separator | See docs |

## Other

| Function | Description | Example |
|----------|-------------|---------|
| `expandtabs(arr, tabsize)` | Expand tabs | `expandtabs(["a\tb"], 4)` → `["a   b"]` |
| `translate(arr, table)` | Char translation | See docs |
| `encode(arr)` | Encode (no-op in C++) | Returns copy |
| `decode(arr)` | Decode (no-op in C++) | Returns copy |
| `mod(arr, vals)` | String formatting | `mod(["%s"], ["hi"])` → `["hi"]` |
| `compare_chararrays(a, b, op)` | Flexible comparison | See docs |

## Common Patterns

### Clean & Normalize
```cpp
auto cleaned = lower(strip(data));
```

### Check & Filter
```cpp
auto valid = isdigit(data);  // bool array
// Use with where() or manual filtering
```

### Format
```cpp
auto names = lower(replace(strip(names), " ", "."));
auto emails = add(names, array(std::vector<std::string>{"@company.com", ...}));
```

### Parse
```cpp
auto fields = split(csv_row, ",");
auto clean_fields = strip(fields);
```

## Return Types

| Operation | Returns |
|-----------|---------|
| String manipulation | `Ndarray<std::string>` |
| Comparison/Testing | `Ndarray<bool>` |
| Search/Count/Length | `Ndarray<int>` |

## Notes

- ⚠️ `split()`, `rsplit()`, `splitlines()` return **flattened** arrays
- ⚠️ Functions in `np::ch` namespace (not `np::char`)
- ⚠️ Not auto-included in `np.hpp` - must `#include "np/char.hpp"`
- ✅ All operations are element-wise
- ✅ All arrays must have matching shapes for binary operations

## See Also

- **Full Examples**: `CHAR_USAGE_EXAMPLES.md`
- **Implementation**: `include/np/char.hpp`
- **Tests**: `tests/test_char.cpp`
- **Status**: `CHAR_IMPLEMENTATION_COMPLETE.md`
