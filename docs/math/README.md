# Math Proofs — Index (dev)

This `docs/math/` folder is the **per-module expansion** of `../MATH_PROOFS.md` (master). Each file proves a block of the 712 routines + its `dev` micro-opts.

**Method:** Every `np::` is `numpy-reference/reference/generated/numpy.<func>.html` formula → C++ loop; proof = pointwise equality + Lemma 0.3 (contiguous fast path).

| File | Module(s) | Routines | Hot file:line |
|------|-----------|----------|---------------|
| [00_overview](00_overview.md) | Preliminaries, Lemma 0.3, notation | — | `ndarray.hpp:3116,3479` |
| [01_ndarray](01_ndarray.md) | Core `ndarray` `is_contiguous`, `_flat_logical`, `_for_each`, `at` | 1 engine | `ndarray.hpp:3116,3479,3315,3539` |
| [02_creation](02_creation.md) | `creation` 42: `arange`, `linspace`, `eye` | 42 | `creation.hpp:65` |
| [03_manipulation](03_manipulation.md) | `manipulation` 45: `copyto`, `split`, `atleast` | 45 | `manipulation.hpp:1980,728` |
| [04_linalg](04_linalg.md) | `linalg` 44: `dot` BLOCK, `norm`, `cross`, `einsum` | 44 | `linalg.hpp:2669,1630,3322,3837` |
| [05_fft](05_fft.md) | `fft` 18: radix2, Bluestein, shifts | 18 | `fft/fft_core.hpp:244` |
| [06_random](06_random.md) | `random` 50: `Generator`, `_fill`, `shuffle` | 50 | `random.hpp:64,1277,161` |
| [07_statistics_sorting](07_statistics_sorting.md) | `statistics` 19 + `sorting` 15 | 34 | `logic.hpp:590,669`, `indexing.hpp:576` |
| [08_optimizations](08_optimizations.md) | All micro-opts synthesis, hash>64, week arithmetic, WASM/RVV | — | `simd.hpp:983`, `threadpool.hpp:236` |

**How to read:** Each proof states **Claim**, **NumPy spec** (LaTeX), **C++ loop**, **Equivalence argument**, **Complexity**, **Optimization proof** (fast = slow via Lemma 0.3).

**Verification:** `cmake --build build -j8 && ctest --test-dir build` → `22/22` after each `dev` micro-opt (`f7b2653..12115ad`). `grep -n "is_contiguous" include/np/ndarray.hpp` etc.

See `../MATH_PROOFS.md` for the 26-group master, `../PERFORMANCE.md` for bench deltas.
