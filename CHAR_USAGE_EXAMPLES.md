# NumPy char Module Usage Examples

Complete examples for using the `np::ch` (numpy.char) module for string array operations.

## Basic Usage

```cpp
#include "np/char.hpp"
#include <iostream>
#include <vector>
#include <string>

int main() {
    using namespace np;
    using namespace np::ch;
    
    // Create string arrays
    auto names = array(std::vector<std::string>{"alice", "bob", "charlie"});
    auto greetings = array(std::vector<std::string>{"hello", "hi", "hey"});
    
    // String concatenation
    auto messages = add(greetings, array(std::vector<std::string>{" ", " ", " "}));
    messages = add(messages, names);
    // Result: ["hello alice", "hi bob", "hey charlie"]
    
    // Convert to uppercase
    auto upper_names = upper(names);
    // Result: ["ALICE", "BOB", "CHARLIE"]
    
    return 0;
}
```

## String Operations

### Case Conversion

```cpp
auto text = array(std::vector<std::string>{"Hello World", "PYTHON", "c++"});

// Lowercase
auto lower_text = lower(text);
// ["hello world", "python", "c++"]

// Uppercase
auto upper_text = upper(text);
// ["HELLO WORLD", "PYTHON", "C++"]

// Title case
auto title_text = title(text);
// ["Hello World", "Python", "C++"]

// Capitalize (first char only)
auto cap_text = capitalize(text);
// ["Hello world", "Python", "C++"]

// Swap case
auto swap_text = swapcase(text);
// ["hELLO wORLD", "python", "C++"]
```

### Padding and Alignment

```cpp
auto words = array(std::vector<std::string>{"cat", "elephant", "dog"});

// Left justify
auto left = ljust(words, 10, '-');
// ["cat-------", "elephant--", "dog-------"]

// Right justify
auto right = rjust(words, 10, '-');
// ["-------cat", "--elephant", "-------dog"]

// Center
auto centered = center(words, 10, '*');
// ["***cat****", "*elephant*", "***dog****"]

// Zero fill (for numbers)
auto numbers = array(std::vector<std::string>{"42", "-7", "100"});
auto zfilled = zfill(numbers, 5);
// ["00042", "-0007", "00100"]
```

### Trimming

```cpp
auto messy = array(std::vector<std::string>{"  hello  ", "\tworld\n", "  test"});

// Strip both sides
auto clean = strip(messy);
// ["hello", "world", "test"]

// Strip left only
auto lclean = lstrip(messy);
// ["hello  ", "world\n", "test"]

// Strip right only
auto rclean = rstrip(messy);
// ["  hello", "\tworld", "  test"]

// Custom characters
auto urls = array(std::vector<std::string>{"https://example.com", "http://test.org"});
auto domains = lstrip(urls, "https://");
// Still has "https://" - need to handle separately or use replace
```

### String Manipulation

```cpp
auto text = array(std::vector<std::string>{"hello world", "foo bar baz"});

// Replace substring
auto replaced = replace(text, "o", "0");
// ["hell0 w0rld", "f00 bar baz"]

// Replace with limit
auto replaced_once = replace(text, "o", "0", 1);
// ["hell0 world", "f00 bar baz"]

// Repeat strings
auto words = array(std::vector<std::string>{"ha", "ho"});
Ndarray<int> counts = empty<int>(std::vector<int>{2});
counts.data() = {3, 2};
auto repeated = multiply(words, counts);
// ["hahaha", "hoho"]

// Expand tabs
auto tabbed = array(std::vector<std::string>{"a\tb\tc", "x\ty"});
auto expanded = expandtabs(tabbed, 4);
// ["a   b   c", "x   y"]
```

## Comparison Operations

```cpp
auto a = array(std::vector<std::string>{"apple", "banana", "cherry"});
auto b = array(std::vector<std::string>{"apple", "berry", "cherry"});

// Equality
auto eq = equal(a, b);
// [true, false, true]

// Inequality  
auto ne = not_equal(a, b);
// [false, true, false]

// Lexicographic comparison
auto lt = less(a, b);
// [false, true, false]

auto gt = greater(a, b);
// [false, false, false]

// Flexible comparison
auto cmp = compare_chararrays(a, b, "<", false);
// Returns int array: 0 or 1 for each comparison
```

## Search and Information

```cpp
auto text = array(std::vector<std::string>{"hello world", "python programming"});

// Find substring
auto pos = find(text, "o");
// [4, 4] (first occurrence)

// Find from right
auto rpos = rfind(text, "o");
// [7, 4] (last occurrence)

// Count occurrences
auto cnt = count(text, "o");
// [2, 2]

// Check start/end
auto starts = startswith(text, "hello");
// [true, false]

auto ends = endswith(text, "ing");
// [false, true]

// String lengths
auto lengths = str_len(text);
// [11, 19]
```

## Testing String Properties

```cpp
auto test = array(std::vector<std::string>{"abc", "ABC", "123", "abc123", "   "});

// Check all alphabetic
auto alpha = isalpha(test);
// [true, true, false, false, false]

// Check all digits
auto digit = isdigit(test);
// [false, false, true, false, false]

// Check alphanumeric
auto alnum = isalnum(test);
// [true, true, true, true, false]

// Check lowercase
auto lower_check = islower(test);
// [true, false, false, true, false]

// Check uppercase
auto upper_check = isupper(test);
// [false, true, false, false, false]

// Check whitespace
auto space = isspace(test);
// [false, false, false, false, true]

// Check title case
auto titles = array(std::vector<std::string>{"Hello World", "hello world", "HELLO"});
auto is_title = istitle(titles);
// [true, false, false]
```

## Split and Partition

```cpp
auto sentences = array(std::vector<std::string>{"a b c", "x,y,z"});

// Split on whitespace
auto words1 = split(sentences);
// Flattened: ["a", "b", "c", "x,y,z"]

// Split on separator
auto parts = split(array(std::vector<std::string>{"a,b,c"}), ",");
// ["a", "b", "c"]

// Split with limit
auto limited = split(array(std::vector<std::string>{"a,b,c,d"}), ",", 2);
// ["a", "b", "c,d"]

// Partition (split into 3 parts: before, sep, after)
auto url = array(std::vector<std::string>{"https://example.com"});
auto parts3 = partition(url, "://");
// ["https", "://", "example.com"] (3 elements)

// Partition from right
auto path = array(std::vector<std::string>{"/home/user/file.txt"});
auto rparts = rpartition(path, "/");
// ["/home/user", "/", "file.txt"]

// Split lines
auto multiline = array(std::vector<std::string>{"line1\nline2\nline3"});
auto lines = splitlines(multiline);
// ["line1", "line2", "line3"]
```

## Practical Examples

### Data Cleaning

```cpp
// Clean survey responses
auto responses = array(std::vector<std::string>{
    "  Yes  ", " no ", "YES", "  No  "
});

// Normalize
auto cleaned = strip(responses);
auto normalized = lower(cleaned);
// ["yes", "no", "yes", "no"]

// Check for specific values
auto is_yes = equal(normalized, array(std::vector<std::string>{"yes", "yes", "yes", "yes"}));
```

### URL Processing

```cpp
auto urls = array(std::vector<std::string>{
    "https://example.com/page",
    "http://test.org/index.html"
});

// Extract protocol
auto has_https = startswith(urls, "https://");
// [true, false]

// Simple domain extraction (simplified)
auto no_protocol = replace(urls, "https://", "");
no_protocol = replace(no_protocol, "http://", "");
// ["example.com/page", "test.org/index.html"]
```

### Format Email Addresses

```cpp
auto names = array(std::vector<std::string>{"Alice Smith", "Bob Jones"});
auto domain = "@example.com";

// Create email addresses
auto lowercase_names = lower(names);
auto no_spaces = replace(lowercase_names, " ", ".");
auto emails = add(no_spaces, array(std::vector<std::string>{domain, domain}));
// ["alice.smith@example.com", "bob.jones@example.com"]
```

### Validate Input

```cpp
auto user_inputs = array(std::vector<std::string>{"12345", "abc", "67890", "xyz"});

// Check if all are digits (valid numeric IDs)
auto valid = isdigit(user_inputs);
// [true, false, true, false]

// Filter valid inputs (pseudo-code - would need where() function)
// auto valid_ids = user_inputs[valid];
```

### Parse CSV-like Data

```cpp
auto csv_row = array(std::vector<std::string>{"John,25,USA"});
auto fields = split(csv_row, ",");
// ["John", "25", "USA"]

// Clean field whitespace
auto cleaned_fields = strip(fields);
```

## Advanced Patterns

### Chaining Operations

```cpp
auto raw_data = array(std::vector<std::string>{"  HELLO  ", "  WORLD  "});

// Chain: strip -> lowercase -> capitalize
auto processed = capitalize(lower(strip(raw_data)));
// ["Hello", "World"]
```

### Conditional Processing

```cpp
auto tags = array(std::vector<std::string>{"#python", "#C++", "#javascript"});

// Check which start with #
auto has_hash = startswith(tags, "#");
// [true, true, true]

// Remove # (replace with empty string)
auto clean_tags = replace(tags, "#", "");
// ["python", "C++", "javascript"]

// Normalize case
auto norm_tags = lower(clean_tags);
// ["python", "c++", "javascript"]
```

## Performance Tips

1. **Minimize Copies**: Operations return new arrays, so avoid unnecessary intermediate steps
2. **Batch Operations**: Process multiple strings at once rather than looping in user code
3. **Reuse Shape**: When creating multiple arrays of the same shape, operations are more efficient

## Notes

1. **Flattened Results**: `split()`, `rsplit()`, and `splitlines()` return flattened arrays (not object arrays of lists like Python NumPy)
2. **No Unicode Support**: C++ `std::string` operations work on bytes; full Unicode support would require additional libraries
3. **Namespace**: Functions are in `np::ch` namespace (not `np::char` which is a C++ keyword)
4. **Element-wise**: All operations work element-wise on `Ndarray<std::string>`

## See Also

- `include/np/char.hpp` - Full API documentation
- `tests/test_char.cpp` - Complete test examples
- `numpy-reference/reference/routines.char.html` - NumPy documentation reference
