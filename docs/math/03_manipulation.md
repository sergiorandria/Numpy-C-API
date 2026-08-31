# 03 — Manipulation (45 routines)

## `copyto:1980` — `manipulation.hpp`

**Spec:** `dst[idx]=src_broadcast[idx]` if `where[idx]` (or always).

**Dev:** `if same shape && is_contiguous() && !where [[likely]]` → `memcpy(dst.data(), src.data(), n*sizeof(T))` with `__restrict`. By Lemma 0.3, `memcpy` copies `offset+i` sequence exactly, so equals `Odometer` per-element loop. Masked variant `if (mask[i]) ap[i]=vp[i]` when both contiguous — pointwise same.

## `atleast_1d/2d/3d:1606`

**Spec:** `ndim>=2 ? view : reshape([1,N])` etc. Dev: `return arr` (shared_ptr) vs `copy()` — zero-copy view, same shape/strides, correct per `reshape` sharing.

## `split:703` / `array_split:852` / `block:1832`

**Spec:** `split` at indices `split_points = {0, indices..., n}` → `arr[ start:end ]` slices via `Odometer`. Dev: `reserve(sections)` / `reserve(n+2)` — alloc hint only, slices via `src_idx/dst_idx` loops same.

## `pad`, `transpose`, `moveaxis`, `broadcast_to`

**Proof:** `Odometer` enumeration of `out_shape` with `src_idx` mapping via `broadcast` rules — same as NumPy. `pad` constant mode `memcpy` interior when contiguous.

## `unique` etc

Delegates to `logic::unique_all:730` — proven there.

## `as_strided`, `sliding_window_view`

**Spec:** `as_strided:2395` shares `data` with new `shape/strides`; `sliding_window_view` via `Odometer` copy — same as `numpy.lib.stride_tricks`.

Complexity: `copyto` O(n) `memcpy` vs O(n) `Odometer`; `split` O(n).
