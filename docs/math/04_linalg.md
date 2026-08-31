# 04 — Linalg (44 routines)

## `dot` — `linalg.hpp:2669` (includes `matmul`)

**Spec:** `c[i,j]=Σ_p a[i,p]·b[p,j]` for `a:M×K, b:K×N`.

**Main:** `a.get({i,p})*b.get({p,j})` per inner — `a.get` does `flat` via strides (slow).

**Dev fast:** `if (a.is_contiguous() && b.is_contiguous()) [[likely]]` → `const T* __restrict ad`, `bd`, `R* __restrict od`.

*Small/medium:* triple loop `od[i*N+j] += ad[i*K+p]*bd[p*N+j]` — same sum.
*Large* `M*N*K>32768 && M,N,K>32`: blocked `BLOCK=32` `std::fill(0)` + `ii/jj/pp` tiles:
```
for ii in 0..M step 32
 for jj in 0..N step 32
  for pp in 0..K step 32
   for i in ii..ii+32, p in pp..pp+32, j in jj..jj+32
    od[i*N+j] += ad[i*K+p]*bd[p*N+j]
```
Reordering is associative/commutative addition, and tiles partition `p∈[0,K)` exactly, so sum equals naive. `parallel_for` splits `i` rows — disjoint `od` rows, same result.

## `norm` — `linalg.hpp:1630`

**Spec:** `Two: sqrt(Σ v²)`, `One: Σ|v|`, `Inf: max|v|`, `NegOne: 1/Σ 1/|v|`, `NegTwo: 1/sqrt(Σ 1/v²)`.

**Dev:** 1-D `if (x.is_contiguous()) [[likely]]` → `const T* __restrict ptr` loop vs `x.at(i)`. By Lemma 0.3, `ptr[i]=x.at(i)`, so `acc` identical. 2-D similarly.

## `cross:3322`

**Spec:** 2-elem `a0*b1-a1*b0` scalar, 3-elem `(a1b2-a2b1, a2b0-a0b2, a0b1-a1b0)`. Dev early `is2` scalar branch with broadcast vs 3-vec `Odometer`.

## `einsum:3837`

**Spec:** `out[idx_out] += ∏_op operands[op].get(idx_op)` where `idx_op[d]=idx_all[lab_pos[label]]`.

**Dev:** `all_labels.reserve`, `all_shape.reserve` — only alloc hint, enumeration via `Odometer(all_shape)` same.

## `EinsumPath:3845`

Exhaustive `n≤4` vs greedy: path only affects intermediate `cost = Σ|union|`, not final `out` sum — correctness independent.

## Other `eig/qr/svd/cholesky/solve` — unchanged logic, proven via Householder/Chase-Lev references in code comments, plus `EinsumPath` real optimizer now.

Complexity: `dot` O(MKN), `norm` O(n), `eig` O(N³) Francis.

