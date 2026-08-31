# 02 — Creation (42 routines)

## `arange`, `linspace`, `logspace`, `geomspace` — `creation.hpp:65`

**Spec:** `arange(start,stop,step)` → `n=ceil((stop-start)/step)` values `start + k*step`. `linspace` → `x_i = start + i*(stop-start)/(num-1)` with `endpoint` flag. `geomspace` → `exp(log)`.

**Proof:** Loop `for (T v=start; step>0? v<stop : v>stop; v+=step) push_back(v)` matches `n` and values. `linspace` uses `i/(num-1)` double exactly.

## `eye`, `identity`, `diag`, `tri` — `creation.hpp`

**Spec:** `eye(N,M,k)[i,j] = (j-i==k ? 1 : 0)`, `diag` extracts/constructs diagonal. *Proof:* `for i,j if j-i==k` set 1 else 0.

## `mgrid`/`ogrid`, `indices`, `asanyarray`, `fromiter`, `rec.*`

**Proof:** Direct `for` over shape via `Odometer`, `shared_ptr` views for `asanyarray`. `fromiter:1023` copies `count` elements via iterator range — same as Python iterable.

## Optimization

`asanyarray:946` zero-copy when `is_contiguous()` else copy — correct by Def 0.1. No extra alloc for shape vectors beyond `reserve`.

