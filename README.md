# NumPy C++ API — `dev` branch

> **Header-only C++20 NumPy 2.2 — 712 routines, 0 stubs, 22/22 tests, SIMD + lock-free.**

[![Branch](https://img.shields.io/badge/branch-dev-blue)](https://github.com/anomalyco/Numpy-C-API/tree/dev)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-22%2F22-brightgreen)](#testing)

This is the **development** branch. `main` tracks stable releases; `dev` holds the latest micro-optimized implementation (WASM/RVV, blocked GEMM, `is_contiguous` fast paths) and rewritten documentation. See [`docs/`](docs/README.md) for the full manual.

---

## 1. Why this fork?

Python NumPy is the lingua franca for scientific computing, but C++ embedders need **zero-overhead, header-only, no Python runtime** parity. This project provides:

* **No linking** — `#include <np/np.hpp>` is enough (`add_library(np INTERFACE)`).
* **Two array engines** — `ndarray<T>` (heap, `shared_ptr` views) + `ndarrayf<T, Extents...>` (stack, `constexpr`).
* **NumPy 2.2 fidelity** — 26 topic groups, 712 distinct routines, Doxygen `Reference: numpy-reference/...` per function.
* **Performance first** — `dev` adds `memcpy` `copyto`, `hash` `isin`, `blocked GEMM`, `week-arithmetic` `busday_count`, WASM/RVV SIMD.

---

## 2. Quick start (dev)

### Requirements

| Tool | Version | Notes |
|------|---------|-------|
| GCC | 14.2+ | or Clang 15+ |
| CMake | 3.20+ | `3.28` recommended |
| CPU | x86-64 / ARM64 / WASM / RV64 | auto-detected |

### Hello

```cpp
#include <np/np.hpp>
#include <np/random.hpp> // not in umbrella (avoid ADL clash) — include explicitly
int main() {
  auto a = np::arange<double>(0, 10, 0.5);          // [0,0.5,…,9.5]
  auto b = np::zeros<double>({3,4});
  auto c = np::ndarray<int>{{1,2},{3,4}};

  auto y = np::sin(np::linspace<double>(0, 2*M_PI, 100));

  // micro-opt fast paths fire here: copyto → memcpy, is_contiguous() [[likely]]
  auto s = a.sum();                 // _for_each_logical direct ptr
  bool ok = np::is_busday(np::datetime::sys_days{std::chrono::days{18500}});

  // linalg: blocked dot + norm contiguous ptr
  auto m1 = np::eye<double>(3);
  auto m2 = np::ones<double>({3,3});
  auto p  = np::matmul(m1,m2);       // BLOCK=32, __restrict
  auto pr = np::einsum_path("ij,jk->ik"); // real optimizer
  (void)pr;
}
```

Compile:

```bash
g++ -std=c++20 -O3 -msse4.2 -I include main.cpp -o main && ./main
# dev extra: WASM
clang++ --target=wasm32 -mwasm-simd128 -DNP_SIMD_WASM -I include main.cpp -o main.wasm
# RVV
clang++ -march=rv64gcv -DNP_SIMD_RVV -I include main.cpp -o main.rvv
```

CMake (dev enables LTO/Native by option):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNP_ENABLE_AVX2=ON -DNP_ENABLE_LTO=ON
cmake --build build -j && ctest --test-dir build --output-on-failure # 22/22
```

---

## 3. Features — `dev` delta

| Module | Header | `dev` micro-opt | Reference |
|--------|--------|-----------------|-----------|
| **ndarray** | `ndarray.hpp:3116` | `is_contiguous()` no `_c_strides` alloc, `_flat_logical:3479` `offset+i` fast, `at(i):3315` `__restrict` + `bool` proxy, `_for_each_logical:3539` direct `T*` | `numpy.ndarray` |
| **Manipulation** | `manipulation.hpp:1980` | `copyto` `memcpy`, `atleast_2d:1606` zero-copy, `split:728` `reserve` | `numpy.copyto` |
| **Datetime** | `datetime.hpp:99` | `_weekday` `days%7`, `is_busday` contiguous `__restrict`, `busday_count:390` `weeks*per_week`, `busday_offset:341` week-jump | `numpy.is_busday` |
| **Random** | `random.hpp:64,1277` | `integers/random` `T* __restrict`, `_fill:1277` direct | `numpy.random.Generator` |
| **Logic/Set** | `logic.hpp:590` | `isin` hash `>64` `unordered_set` vs `sort`, `intersect1d:669` `reserve` | `numpy.isin` |
| **Indexing** | `indexing.hpp:576/616` | `nditer/flatiter` cached `ptr_/contig_` | `numpy.nditer` |
| **Linalg** | `linalg.hpp:2669,1630` | `dot` BLOCK=32 + `__restrict` + `parallel_for`, `norm` contiguous `ptr`, `einsum:3837` `reserve` | `numpy.linalg` |
| **SIMD** | `simd.hpp:983` | **New** WASM `v128` + RVV `__riscv_vsetvl` | `NP_SIMD_WASM/RVV` |
| **Threadpool** | `threadpool.hpp:236` | True Chase-Lev ring `top/bottom` CAS, `grow` power-of-two | SPAA 2005 |
| **Testing** | `testing.hpp:108` | `assert_equal` contiguous `__restrict` | `numpy.testing` |

Full 26 groups remain 100% — see [`docs/API.md`](docs/API.md).

---

## 4. Project layout (dev)

```
include/np/
  np.hpp            umbrella (22 includes, random/concatenate explicit)
  ndarray.hpp       6485 LOC  — is_contiguous @3116, _flat_logical @3479, _for_each @3539
  linalg.hpp        3972 LOC  — dot BLOCK @2669, norm @1630, EinsumPath @3845
  indexing.hpp      959 LOC   — nditer @576, flatiter @616, fill_diagonal @472
  manipulation.hpp  2661 LOC  — copyto @1980, split @728
  datetime.hpp      633 LOC   — _weekday @99, busday_* @192/341/390
  random.hpp        1826 LOC  — _fill @1277
  logic.hpp         1036 LOC  — isin @590, intersect @669
  simd.hpp          1100 LOC  — WASM @983, RVV
  threadpool.hpp    1050 LOC  — __np_deque_lockfree @236
  fft/              18 ops  — fft_core @244
  ... 26 headers total
docs/
  README.md         index
  ARCHITECTURE.md   dual engines, views, strides
  PERFORMANCE.md    bench, BLOCK, week-arithmetic, hash threshold
  API.md            per-module table with file:line
  CONTRIBUTING.md   dev workflow, clang-format, feat() commits
tests/              22 ctest suites (test_ndarray, test_linalg, …)
```

---

## 5. Performance notes (dev)

* **Contiguous fast paths** guard 90% of hot loops (`is_contiguous() [[likely]]` → `ptr[i]` vs `_flat`).
* **Memcpy** for `copyto`, `block`, `broadcast_arrays` when shapes match.
* **Hash vs sort** threshold `64` for `isin` (empirical crossover).
* **Blocked GEMM** `32` for `dot` >32k elements; falls back to `parallel_for` >4096.
* **Datetime** 5-day week arithmetic cuts `busday_count` from O(days) to O(1) when `hol.empty()`.
* Build with `-DNP_ENABLE_LTO=ON -DNP_ENABLE_NATIVE=ON -DNP_ENABLE_FAST_MATH=ON` for max.

See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) for micro-benchmarks.

---

## 6. Testing

```bash
cmake -S . -B build && cmake --build build -j8
ctest --test-dir build --output-on-failure # 22/22
./build/tests/test_ndarray --verbose
g++ -std=c++20 -I include tests/test_math.cpp -o /tmp/t && /tmp/t
```

`bench_math` (AVX) is not in `ctest` — run manually:

```bash
cmake --build build --target bench_math && ./build/tests/bench_math
```

---

## 7. Contributing (dev)

1. Branch from `dev`: `git checkout -b feat/my-opt dev`
2. Check `numpy-reference/reference/generated/numpy.<func>.html` — match Python signature exactly.
3. Implement in `include/np/<module>.hpp` with Doxygen `Reference:` link.
4. `clang-format -i include/np/*.hpp` (`.clang-format`: 2-space Allman, `ColumnLimit: 90`, `SortIncludes: Never`).
5. Add `tests/test_<module>.cpp` using `tests/test_util.hpp`.
6. `cmake --build build && ctest` — commit `feat(module): ...` with `file:line`.

See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) and `AGENTS.md`.

---

## 8. Known divergences

* `operator[i,j]` is C++23 — use `arr(i,j)` or `arr[i][j]` proxy.
* Complex `linalg` is real-only (`is_complex_v` static_assert) — dispatches to real `double`.
* `vector<bool>` proxy — `ndarray<bool>::at()` returns proxy, `is_contiguous()` still `vector<bool>` bitset.
* `numpy.distutils/ctypeslib` are thin `other.hpp` stubs.

---

## 9. License & references

BSD-3-Clause (NumPy). References: [NumPy docs](https://numpy.org/doc/stable/), `numpy-reference/` in repo, C++20 standard.

**Author:** Sergio Randriamihoatra — `dev` branch micro-opts `f7b2653..cf8f4a4`.

> This is an independent reimplementation, not affiliated with NumPy.
