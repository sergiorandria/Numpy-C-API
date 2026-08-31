# 01 — ndarray Core

## `is_contiguous:3116`

**Claim:** `is_contiguous` equals `strides==_c_strides(shape)`.

*Main:* `strides != _c_strides` allocs `vector<size_t>` per call.
*Dev:* `exp=1; for i=size..0 { if strides[i]!=exp → false; exp*=shape[i]; }` — same predicate, no alloc. `[[unlikely]]` on `offset!=0` or mismatch.

## `_flat:3472`, `_flat_logical:3479`

**Claim:** `_flat(idx) = detail::flat_index(idx,strides,offset)` and `_flat_logical(i)` as above. Dev adds `is_contiguous() [[likely]] → offset+i` fast. Proven in Lemma 0.3.

## `at(i):3315`, `at(i,j):3417`, `operator()(i,j):3391`

**Claim:** `at(i)` requires `shape.size()==1` and `i < shape[0]`, then `data[offset+i*strides[0]]`. Dev: `[[unlikely]]` on throw paths + `if constexpr (is_same_v<T,bool>)` proxy fallback else `T* __restrict d=data.data()`.

**Proof:** Bounds same; `__restrict` does not change `d[flat]` value.

## `_for_each_logical:3539`

**Claim:** Iterates `0..n-1` via `fn(data[flat])`. Dev: `is_contiguous() [[likely]]` → `const T* __restrict p=data.data(); for i<n fn(p[i])` else `Odometer`. By Lemma 0.3, `p[i]=data[flat]` for contiguous, so same call sequence. `bool` specialization `for (auto& v: *data_)` handles `vector<bool>` proxy.

**Corollary:** `sum:3680` `total+=v` via `_for_each_logical` is `Σ a[i]` correct; similarly `mean/prod/all/any`.

## `sum`/`mean`/`var` Complexity

`O(n)` with `n=_numel()`. Contiguous path is `n` loads + `n` adds, no `Odometer` overhead.

## `is_f_contiguous:3126` (Fortran)

Same `exp` loop but `stride=1` forward.

## `data()` access

`T* __restrict` hints tell compiler no alias, enabling auto-vectorization; correctness unchanged.
