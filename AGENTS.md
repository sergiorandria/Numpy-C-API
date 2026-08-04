# AGENTS.md

Guidance for AI agents working on this repository.

## Project

Header-only C++20 NumPy clone. The public API mirrors numpy 2.x semantics, and
the offline docs mirror at `numpy-reference/` is the ground truth for
signatures and behavior (e.g. `numpy-reference/reference/generated/numpy.arange.html`,
`numpy-reference/user/basics.broadcasting.html`).

- Dynamic runtime path: `np::Ndarray` (heap-allocated, runtime shape checks,
  throws `std::invalid_argument` / `np::AxisError`).
- Compile-time-first path: `np::ndarray<T, Extents...>` (stack storage, shape
  encoded in the type, errors via `static_assert` / `requires` clauses).
- All code lives under `include/np/`, everything is `constexpr` where possible.

## Layout

- `include/np/np.hpp` — umbrella header (add new public headers here).
- `include/np/detail/` — implementation details:
  - `expr.hpp` — shape_tag, compile-time broadcasting, lazy expr nodes
  - `math_constexpr.hpp` — constexpr math kernels (namespace `np::detail::math`)
  - `proxy.hpp` — index proxy used by `Ndarray`
- `tests/` — self-contained executables, one per target, no external deps.
- Legacy code was deleted (was: `Matrix/`, `misc/`, `numpy/`, old root headers).

## Build and test

Toolchain: g++ 14.2.0 (MinGW-W64), CMake 3.30.2.

```powershell
cmake -S . -B build
cmake --build build --config Release --clean-first
ctest --test-dir build -C Release --output-on-failure
```

- C++20 (`-std=c++20`), `-Wall -Wextra` must stay warning-free.
- `NP_COMPILED_UNITS=ON` (default) adds the compile-time test units
  (`test_constexpr`, `test_compile_time`). `cmake -S . -B build -DNP_COMPILED_UNITS=OFF`
  builds the 8 runtime tests only.
- For quick single-file iteration (header-only, so `g++ -I include` is enough):
  `g++ -std=c++20 -Wall -Wextra -I include tests/test_ndarray.cpp -o t && .\t`

## Conventions

- Style: 4-space indent, `snake_case` for functions/params, `PascalCase` types,
  Doxygen `/** @brief ... */` comments on public API, `// --- section ---`
  dividers. Do not add comments unless they carry information.
- Reference every public function to its numpy doc page, e.g.
  `// Reference: numpy-reference/reference/generated/numpy.sum.html`.
- Include guard style: `#ifndef NP_<NAME>_HPP` / `#define NP_<NAME>_HPP`.
- Tests use `test::check(cond, "what")` and `test::approx(...)` from
  `tests/test_util.hpp`; return `test::failures() ? 1 : 0`.
- Compile-time test style: `test_constexpr.cpp` asserts via `static_assert`
  only; `test_compile_time.cpp` proves invalid code is rejected using
  `requires`-expression detector concepts + `static_assert(!detector<...>)`.
- New tests must be added to the `NP_TESTS` list in `tests/CMakeLists.txt`
  (split: runtime tests in the base list, compile-time units gated on
  `NP_COMPILED_UNITS`).

## C++20 / toolchain pitfalls (learned the hard way)

- **No `a[i, j]` indexing.** Multi-argument `operator[]` is C++23 (P2128).
  Fixed arrays use `operator()(i, j, ...)` for multi-dim and `operator[](flat)`
  for flat access. Do not add variadic `operator[]`.
- **Parameter packs must be at the end** of the template parameter list
  (GCC hard error). `template <int... E, typename T>` only works when `T` is
  deducible from the call. Hence two overloads for argless creators:
  `np::zeros<2, 3>()` (double) and `np::zeros<int, 2, 3>()`.
- **`std::shift_left` / `std::shift_right` do not exist** in libstdc++ 14.2;
  elementwise `<<`/`>>` use custom functors in `detail::fixed`.
- **Class-scope fold expressions** like `(Extents > 0 && ...)` miscompile in
  GCC; use `std::conjunction_v<std::bool_constant<...>...>` instead.
- **`std::sqrt/std::exp/...` are not constexpr until C++26** (P0533). Use
  `np::detail::math` kernels for constexpr math.
- `shape_tag` pairs: to compare two packs for equality use the same-pack idiom
  `template <int... A> struct same_tag<shape_tag<A...>, shape_tag<A...>>`
  — a two-packs partial specialization is a bug (matches any tags).
- numpy semantics to respect: `np.round` is half-to-even; `concatenate`
  requires equal non-axis-0 extents; `squeeze(Axis)` requires extent == 1;
  `dot(v, m)` of (K,)·(K,M) returns rank-1 (M,), not a matrix.
