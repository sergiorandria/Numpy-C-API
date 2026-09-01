/**
 * @file homology.hpp
 * @brief Simplicial homology, Smith normal form, Betti numbers.
 *
 * Provides header-only, exact-integer homology for finite simplicial complexes:
 *   - `SimplicialComplex` (by-dimension simplex lists + boundary matrices)
 *   - `smith_normal_form` (exact over Z via `np::bigint`, fallback rank)
 *   - `betti_numbers`, `homology_groups`, `euler_characteristic`
 *   - `simplicial_homology` convenience
 *
 * The SNF is computed exactly for 1×1 / 2×2 (via gcd) and for sparse ±1
 * boundary matrices via rank (totally unimodular → invariant factors 1).
 * Larger arbitrary matrices fall back to `linalg::matrix_rank` over Q for
 * Betti numbers; torsion is reported as empty in that fallback.
 *
 * Reference: Hatcher, *Algebraic Topology* Ch.2; Munkres, *Elements of Algebraic
 * Topology*; Cohen, *A Course in Computational Algebraic Number Theory* (SNF).
 * numpy-reference: not applicable (abstract topology).
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_HOMOLOGY_HPP
#define NP_HOMOLOGY_HPP

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

namespace np::homology
{

  // ── SimplicialComplex ───────────────────────────────────────────────────

  /**
   * @brief Finite abstract simplicial complex by dimension.
   *
   * `simplices[d]` holds the `d`-simplices, each as sorted `vector<int>` of
   * vertex indices. `d=0` are vertices, `d=1` edges, etc.
   * Example tetrahedron boundary (S²): 4 vertices, 6 edges, 4 faces.
   */
  struct SimplicialComplex
  {
    std::vector<std::vector<std::vector<int>>> simplices; // index = dim

    SimplicialComplex() = default;
    explicit SimplicialComplex(std::vector<std::vector<std::vector<int>>> s)
        : simplices(std::move(s))
    {
      for (auto& lvl : simplices)
        for (auto& simp : lvl) std::sort(simp.begin(), simp.end());
    }

    NP_NODISCARD int dim() const noexcept
    {
      return static_cast<int>(simplices.size()) - 1;
    }

    NP_NODISCARD std::size_t num_simplices(int d) const noexcept
    {
      if (d < 0 || d >= static_cast<int>(simplices.size()))
        return 0;
      return simplices[d].size();
    }

    /**
     * @brief Boundary matrix `d_d : C_d -> C_{d-1}`.
     * Rows = (d-1)-simplices, cols = d-simplices, entry ±1 if face.
     * For d<=0 or d>dim returns 0×0.
     */
    NP_NODISCARD ndarray<int> boundary_matrix(int d) const
    {
      if (d <= 0 || d >= static_cast<int>(simplices.size()))
        return ndarray<int>::from_data({0, 0}, std::vector<int>{});
      const auto& higher = simplices[d];
      const auto& lower = simplices[d - 1];
      if (higher.empty() || lower.empty())
        return ndarray<int>::from_data(
            {static_cast<int>(lower.size()), static_cast<int>(higher.size())},
            std::vector<int>(lower.size() * higher.size(), 0));

      // map lower simplex -> index
      std::map<std::vector<int>, int> lower_index;
      for (int i = 0; i < static_cast<int>(lower.size()); ++i)
        lower_index[lower[i]] = i;

      std::vector<int> data(lower.size() * higher.size(), 0);
      for (int j = 0; j < static_cast<int>(higher.size()); ++j)
      {
        const auto& s = higher[j];
        for (int k = 0; k <= d; ++k)
        {
          std::vector<int> face = s;
          face.erase(face.begin() + k);
          auto it = lower_index.find(face);
          if (it != lower_index.end())
          {
            int sign = (k % 2 == 0) ? 1 : -1;
            data[it->second * higher.size() + j] = sign;
          }
        }
      }
      return ndarray<int>::from_data(
          {static_cast<int>(lower.size()), static_cast<int>(higher.size())}, std::move(data));
    }

    /**
     * @brief All boundary matrices `d_1 .. d_dim`.
     * `bms[d]` = `d_d` for `d>=1`, `bms[0]` = 0×0 placeholder.
     */
    NP_NODISCARD std::vector<ndarray<int>> boundary_matrices() const
    {
      std::vector<ndarray<int>> out;
      out.reserve(simplices.size());
      out.push_back(ndarray<int>::from_data({0, 0}, std::vector<int>{})); // d0 placeholder
      for (int d = 1; d < static_cast<int>(simplices.size()); ++d)
        out.push_back(boundary_matrix(d));
      return out;
    }
  };

  // ── helpers: bigint gcd ─────────────────────────────────────────────────

  NP_NODISCARD inline bigint bigint_abs(const bigint& x)
  {
    return x < 0 ? -x : x;
  }

  NP_NODISCARD inline bigint bigint_gcd(bigint a, bigint b)
  {
    a = bigint_abs(a);
    b = bigint_abs(b);
    while (b != 0)
    {
      bigint r = a % b;
      a = b;
      b = r;
    }
    return a;
  }

  // ── Smith normal form ───────────────────────────────────────────────────

  /**
   * @brief Smith normal form diagonal for integer matrix `A` (m×n).
   *
   * Returns sorted invariant factors `diag` of length `min(m,n)` where
   * `diag[i] | diag[i+1]` and zeros for rank deficiency. For 1×1 and 2×2
   * the SNF is exact via gcd; for larger ±1 boundary matrices the SNF is
   * `rank` many 1's (totally unimodular). Otherwise rank is computed over Q
   * via `linalg::matrix_rank` and diag = 1's for rank.
   *
   * Reference: https://en.wikipedia.org/wiki/Smith_normal_form
   */
  NP_NODISCARD inline std::vector<bigint> smith_normal_form(const ndarray<int>& A)
  {
    // convert to bigint
    ndarray<bigint> Ab = as_bigint(A);
    return [&]() -> std::vector<bigint> {
      int m = Ab.shape[0], n = Ab.shape[1];
      int k = std::min(m, n);
      std::vector<bigint> diag(k, bigint(0));
      if (k == 0)
        return diag;
      if (k == 1)
      {
        bigint g = 0;
        for (int i = 0; i < m; ++i)
          for (int j = 0; j < n; ++j)
            g = bigint_gcd(g, Ab(i, j));
        diag[0] = bigint_abs(g);
        return diag;
      }
      if (m == 2 && n == 2)
      {
        bigint a = Ab(0, 0), b = Ab(0, 1), c = Ab(1, 0), d = Ab(1, 1);
        bigint g = bigint_gcd(bigint_gcd(bigint_gcd(a, b), c), d);
        if (g == 0)
          return diag;
        bigint det = a * d - b * c;
        bigint absdet = bigint_abs(det);
        // SNF for 2x2: diag[0]=g, diag[1]=|det|/g if det!=0 else 0
        diag[0] = g;
        if (absdet != 0)
          diag[1] = absdet / g;
        else
          diag[1] = 0;
        // ensure divisibility g | diag1 (holds)
        if (diag[0] < 0) diag[0] = -diag[0];
        if (diag[1] < 0) diag[1] = -diag[1];
        if (diag[0] == 0) std::swap(diag[0], diag[1]);
        return diag;
      }
      // General: boundary matrices are ±1, so SNF is 1's for rank
      // Compute rank over Q via double conversion
      int rows = m, cols = n;
      // Convert to double for rank
      std::vector<double> dd(rows * cols);
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          dd[i * cols + j] = static_cast<double>(Ab(i, j).convert_to<long long>());
      // Use linalg::matrix_rank via double ndarray
      ndarray<double> Ad = ndarray<double>::from_data({rows, cols}, std::move(dd));
      int r = linalg::matrix_rank(Ad);
      for (int i = 0; i < r && i < k; ++i) diag[i] = bigint(1);
      return diag;
    }();
  }

  NP_NODISCARD inline std::vector<bigint> smith_normal_form(const ndarray<bigint>& A)
  {
    int m = A.shape[0], n = A.shape[1];
    int k = std::min(m, n);
    std::vector<bigint> diag(k, bigint(0));
    if (k == 0) return diag;
    if (k == 1)
    {
      bigint g = 0;
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) g = bigint_gcd(g, A(i, j));
      diag[0] = bigint_abs(g);
      return diag;
    }
    if (m == 2 && n == 2)
    {
      bigint a = A(0, 0), b = A(0, 1), c = A(1, 0), d = A(1, 1);
      bigint g = bigint_gcd(bigint_gcd(bigint_gcd(a, b), c), d);
      if (g == 0) return diag;
      bigint det = a * d - b * c;
      diag[0] = bigint_abs(g);
      if (det != 0) diag[1] = bigint_abs(det) / diag[0];
      return diag;
    }
    // fallback rank
    ndarray<double> Ad({m, n});
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) Ad(i, j) = static_cast<double>(A(i, j).convert_to<long long>());
    int r = linalg::matrix_rank(Ad);
    for (int i = 0; i < r && i < k; ++i) diag[i] = bigint(1);
    return diag;
  }

  // ── Betti numbers & homology ────────────────────────────────────────────

  struct HomologyGroup
  {
    int betti = 0;
    std::vector<bigint> torsion; // invariant factors >1
    std::string to_string() const
    {
      std::string s = "Z^" + std::to_string(betti);
      if (!torsion.empty())
      {
        s += " + ";
        for (size_t i = 0; i < torsion.size(); ++i)
        {
          if (i) s += " + ";
          s += "Z/" + torsion[i].convert_to<std::string>() + "Z";
        }
      }
      return s;
    }
  };

  /**
   * @brief Betti numbers for a chain complex given by boundary matrices.
   *
   * `bms[d]` = `d_d` (d>=1), `bms[0]` placeholder, `n_d` = `bms[d].shape[1]` for d>=1,
   * `n_0` = `bms[1].shape[0]`. Over Q: `betti_d = n_d - rank(d_d) - rank(d_{d+1})`.
   */
  NP_NODISCARD inline std::vector<int> betti_numbers(const std::vector<ndarray<int>>& bms)
  {
    int D = static_cast<int>(bms.size()) - 1;
    if (D < 0) return {};
    std::vector<int> n(D + 1, 0);
    for (int d = 0; d <= D; ++d)
    {
      if (d == 0)
        n[0] = (D >= 1) ? bms[1].shape[0] : 0;
      else
        n[d] = bms[d].shape[1];
    }
    // ranks over Q
    std::vector<int> rank(D + 1, 0);
    for (int d = 1; d <= D; ++d)
    {
      if (bms[d].size() == 0) continue;
      // convert to double for rank
      int m = bms[d].shape[0], r = bms[d].shape[1];
      std::vector<double> dd(m * r);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < r; ++j) dd[i * r + j] = static_cast<double>(bms[d](i, j));
      ndarray<double> Ad = ndarray<double>::from_data({m, r}, std::move(dd));
      rank[d] = linalg::matrix_rank(Ad);
    }
    std::vector<int> betti(D + 1, 0);
    for (int d = 0; d <= D; ++d)
    {
      int rd = (d <= D) ? rank[d] : 0;
      int rd1 = (d + 1 <= D) ? rank[d + 1] : 0;
      betti[d] = n[d] - rd - rd1;
      if (betti[d] < 0) betti[d] = 0;
    }
    return betti;
  }

  NP_NODISCARD inline std::vector<int> betti_numbers(const SimplicialComplex& K)
  {
    return betti_numbers(K.boundary_matrices());
  }

  NP_NODISCARD inline std::vector<HomologyGroup>
  homology_groups(const std::vector<ndarray<int>>& bms)
  {
    auto betti = betti_numbers(bms);
    std::vector<HomologyGroup> out(betti.size());
    for (size_t i = 0; i < betti.size(); ++i) out[i].betti = betti[i];
    // Torsion from SNF of d_{d+1}: invariant factors >1
    for (size_t d = 0; d + 1 < bms.size(); ++d)
    {
      if (bms[d + 1].size() == 0) continue;
      auto diag = smith_normal_form(bms[d + 1]);
      for (auto &v : diag)
        if (v > 1) out[d].torsion.push_back(v);
    }
    return out;
  }

  NP_NODISCARD inline std::vector<HomologyGroup> homology_groups(const SimplicialComplex& K)
  {
    return homology_groups(K.boundary_matrices());
  }

  NP_NODISCARD inline int euler_characteristic(const SimplicialComplex& K)
  {
    int e = 0;
    for (int d = 0; d < static_cast<int>(K.simplices.size()); ++d)
    {
      int nd = static_cast<int>(K.simplices[d].size());
      e += (d % 2 == 0) ? nd : -nd;
    }
    return e;
  }

  NP_NODISCARD inline int euler_characteristic(const std::vector<ndarray<int>>& bms)
  {
    auto betti = betti_numbers(bms);
    int e = 0;
    for (size_t i = 0; i < betti.size(); ++i) e += (i % 2 == 0) ? betti[i] : -betti[i];
    return e;
  }

  // ── Convenience constructors ────────────────────────────────────────────

  NP_NODISCARD inline SimplicialComplex make_simplex(int n_vertices)
  {
    // single (n_vertices-1)-simplex plus all faces (power set)
    std::vector<std::vector<std::vector<int>>> s(n_vertices);
    std::vector<int> verts(n_vertices);
    std::iota(verts.begin(), verts.end(), 0);
    // generate all subsets
    for (int mask = 1; mask < (1 << n_vertices); ++mask)
    {
      std::vector<int> face;
      for (int i = 0; i < n_vertices; ++i)
        if (mask & (1 << i)) face.push_back(i);
      int d = static_cast<int>(face.size()) - 1;
      s[d].push_back(face);
    }
    for (auto& lvl : s) std::sort(lvl.begin(), lvl.end());
    return SimplicialComplex{s};
  }

  NP_NODISCARD inline SimplicialComplex circle_complex()
  {
    // triangle boundary: 3 vertices, 3 edges
    return SimplicialComplex{{{{0}, {1}, {2}}, {{0, 1}, {1, 2}, {0, 2}}, {}}};
  }

  NP_NODISCARD inline SimplicialComplex sphere_tetrahedron()
  {
    // boundary of tetrahedron: 4 vertices, 6 edges, 4 faces (hollow)
    return SimplicialComplex{
        {{{0}, {1}, {2}, {3}},
         {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}},
         {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}},
         {}}};
  }

} // namespace np::homology

#endif // NP_HOMOLOGY_HPP
