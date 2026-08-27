# Development Guidelines

This document describes workflow, style, and testing expectations for contributors.

## Workflow
- Check `README.md` and `include/np/*.hpp` for existing API before adding new functions.
- Match NumPy reference signatures exactly (see `numpy-reference/reference/generated/` if available).
- Implement in appropriate header (`creation.hpp`, `math.hpp`, `logic.hpp`, `statistics.hpp`, `linalg.hpp`, `fft/`, `io.hpp`, `char.hpp`, `polynomial.hpp`).
- Add Doxygen comments with `Reference:` link.
- Ensure header is included in `np.hpp` if it is part of the public API.

## Style
- `snake_case` for functions, `PascalCase` for types, 2-space indent, Allman braces (`BreakBeforeBraces: Allman`).
- Format with `clang-format` using `.clang-format` at repo root: `clang-format -i include/np/*.hpp`.
- Keep `UseTab: Never`, `ColumnLimit: 90`, `SortIncludes: Never`.

## Testing
- Add a test file `tests/test_<module>.cpp` using `tests/test_util.hpp` (`test::check`, `test::approx`, `test::approx_c`).
- Register in `tests/CMakeLists.txt` `NP_TESTS`.
- Build and run: `cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`.

## Commits
- One commit per logical task, message `feat(module): ...` with file:line references where helpful.
- Do not commit `build/`, `CMakeFiles/`, `DartConfiguration.tcl`, `CMakeCache.txt`.

## Notes
- `Matrix<T>` is legacy; prefer `ndarray<T>` + `linalg.hpp` for new code.
- `np::ch` (char) helpers operate on `ndarray<std::string>`.
