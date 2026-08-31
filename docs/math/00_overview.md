# 00 — Overview: Model, Notation, Lemma 0.3

## Notation

* `a : ndarray<T>` with `n = _numel() = ∏ shape`, `data: vector<T>`, `strides: vector<size_t>`, `offset`.
* `a[i0,…,ik-1] = data[offset + Σ idx_j·strides[j]]` (C-order).
* `_c_strides(shape)[j] = ∏_{t>j} shape[t]`, `stride[k-1]=1`.
* `flat(i)` for linear `i ∈ [0,n)` via `i → coords → flat` using division by `shape`.

## Lemma 0.3 — Contiguous flat equivalence

**Definition (Contiguous, `ndarray.hpp:3116`):**
```
is_contiguous ⇔ offset==0 ∧ strides==_c_strides(shape) ∧ data.size()≥n
```
New dev: `exp` loop, no alloc, `[[unlikely]]`.

**Definition (`_flat_logical:3479`):**
```
if shape.empty()||i==0 → offset
else if is_contiguous() [[likely]] → offset+i
else flat = offset + Σ ( (i / ∏_{t>j} shape[t]) % shape[j] ) * strides[j]
```
Old: `_shape_u()` vector alloc + `Odometer`.

**Lemma:** For contiguous `a`, `data[_flat_logical(i)] = data[offset+i]`.

*Proof.* Contiguous strides are C-order, so `i`'s mixed-radix expansion `coord_j = (i / stride_j) % shape[j]` satisfies `i = Σ coord_j·stride_j`. Then `flat = offset + Σ coord_j·strides[j] = offset+i`. Non-contiguous branch computes same sum directly without alloc, so equality holds. ∎

**Corollary (Fast path correctness).** Any loop

```cpp
if (is_contiguous() [[likely]]) { const T* p=data.data(); for i<n p[i] } else { Odometer...
```

computes same `data[flat]` sequence, hence any `np::` that delegates to `_for_each_logical` (e.g. `sum:3680`, `mean`) is correct under micro-opt.

**Branch hints** `[[likely]]`/`[[unlikely]]` are performance hints only.

**Reserve/BLOCK** (`split:728` `reserve`, `dot:2669` `BLOCK=32`) do not affect values, only allocs — trivially correct.

This lemma is the single proof for **all** contiguous fast paths below (`copyto`, `is_busday`, `dot`, `norm`, `isin`, `nditer`, `_for_each_logical`).
