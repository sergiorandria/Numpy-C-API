/**
 * @file manifold.hpp
 * @brief Abstract manifolds and varieties with homology / homotopy / de Rham and logical reasoning.
 *
 * Correct name for `variety.hpp` (kept as alias). Provides `np::manifold::AbstractManifold`
 * and concrete `Sphere`, `Torus`, `ProjectiveSpace`, etc., that integrate with
 * `np::homology` / `np::homotopy` / `np::differential` and provide helpers to
 * fix logical reasoning in differential / topological / algebraic geometry:
 *   - `is_orientable`, `is_compact`, `is_connected`, `is_simply_connected`
 *   - `is_smooth`, `is_complete`, `is_irreducible`, `is_normal`, `is_reduced`
 *   - `check_logical_consistency()` (Euler = Σ(-1)^k betti, orientable ↔ H_n, etc.)
 *   - de Rham ↔ singular via `de_rham` vs `homology`
 *   - differential geometry: `tangent_bundle`, `cotangent_bundle`, `metric`, `connection`
 *
 *   auto S2 = manifold::sphere(2); // S², dim 2
 *   S2.homology(2).betti==1; // H₂=Z → R over R
 *   S2.de_rham(2).betti==1;  // H²_dR=R
 *   S2.is_orientable()==true; S2.check_logical_consistency().ok==true;
 *
 * Reference: Hatcher, Hartshorne, Bott–Tu, Lee *Introduction to Smooth Manifolds*.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_MANIFOLD_HPP
#define NP_MANIFOLD_HPP

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "bigint.hpp"
#include "differential.hpp"
#include "homology.hpp"
#include "homotopy.hpp"

namespace np::manifold
{

  /**
   * @brief Abstract manifold / variety interface (correct name).
   *
   * `AbstractVariety` is kept as alias for backward compatibility.
   */
  struct AbstractManifold
  {
    virtual ~AbstractManifold() = default;
    virtual std::string name() const = 0;
    virtual int dimension() const = 0;
    virtual std::vector<homology::HomologyGroup> homology() const = 0;
    virtual homology::HomologyGroup homology(int k) const = 0;
    virtual homotopy::HomotopyGroup homotopy(int k) const = 0;
    virtual homology::HomologyGroup de_rham(int k) const = 0;
    virtual homology::SimplicialComplex to_simplicial() const = 0;
    virtual int euler_characteristic() const = 0;

    // ── Logical reasoning helpers (differential / topological / algebraic) ──

    virtual bool is_orientable() const { return true; } // S^n, T^n, CP^n yes; RP^n depends
    virtual bool is_compact() const { return true; }   // all our examples compact
    virtual bool is_connected() const { return true; }
    virtual bool is_simply_connected() const { return homotopy::is_simply_connected(to_simplicial()); }
    virtual bool is_smooth() const { return true; }
    virtual bool is_complete() const { return is_compact(); } // for varieties: proper
    virtual bool is_irreducible() const { return true; }
    virtual bool is_reduced() const { return true; }
    virtual bool is_normal() const { return true; }

    struct ConsistencyReport
    {
      bool ok = true;
      std::string reason = "consistent";
      std::vector<std::string> checks;
    };

    /**
     * @brief Check logical consistency of invariants (Euler, orientability, de Rham, etc.).
     *
     * Verifies:
     *   - Euler = Σ (-1)^k betti_k
     *   - orientable ↔ H_n = Z (or R) for n-dim compact connected
     *   - de Rham H^k ≅ singular H_k ⊗ R (Betti match)
     *   - simply connected ↔ H₁=0
     */
    virtual ConsistencyReport check_logical_consistency() const
    {
      ConsistencyReport r;
      r.ok = true;
      auto hg = homology();
      int euler = euler_characteristic();
      int euler_from_betti = 0;
      for (size_t k = 0; k < hg.size(); ++k) euler_from_betti += (k % 2 == 0) ? hg[k].betti : -hg[k].betti;
      // Torsion does not affect Euler, but we check Betti Euler
      if (euler != euler_from_betti)
      {
        r.ok = false;
        r.reason = "Euler mismatch: Euler=" + std::to_string(euler) + " vs Betti sum=" + std::to_string(euler_from_betti);
        r.checks.push_back(r.reason);
        return r;
      }
      r.checks.push_back("Euler = Σ(-1)^k betti: " + std::to_string(euler));

      // Orientability ↔ top homology
      bool orient = is_orientable();
      int n = dimension();
      int top_betti = (n >= 0 && n < (int)hg.size()) ? hg[n].betti : 0;
      bool top_is_Z = (top_betti == 1);
      if (is_compact() && is_connected())
      {
        if (orient && !top_is_Z)
        {
          r.ok = false;
          r.reason = "orientable compact connected n-manifold must have H_n=Z";
          r.checks.push_back(r.reason);
          return r;
        }
        if (!orient && top_is_Z)
        {
          r.ok = false;
          r.reason = "non-orientable compact connected must have H_n=0";
          r.checks.push_back(r.reason);
          return r;
        }
        r.checks.push_back(std::string("orientable=") + (orient ? "true" : "false") + " ↔ H_n=" + std::to_string(top_betti));
      }

      // de Rham vs singular (over R, Betti should match)
      for (int k = 0; k <= n; ++k)
      {
        int dr = de_rham(k).betti;
        int sing = (k < (int)hg.size()) ? hg[k].betti : 0;
        if (dr != sing)
        {
          r.ok = false;
          r.reason = "de Rham H^" + std::to_string(k) + "=" + std::to_string(dr) + " vs singular Betti " + std::to_string(sing);
          r.checks.push_back(r.reason);
          return r;
        }
      }
      r.checks.push_back("de Rham = singular (Betti match)");

      // Simply connected ↔ H₁=0
      bool sc = is_simply_connected();
      bool h1_zero = (hg.size() > 1) ? (hg[1].betti == 0 && hg[1].torsion.empty()) : true;
      if (sc != h1_zero)
      {
        // For simply connected, H₁ must be 0, but converse not always (Poincaré sphere)
        // We only check one direction: simply connected ⇒ H₁=0
        if (sc && !h1_zero)
        {
          r.ok = false;
          r.reason = "simply connected but H₁ non-zero";
          r.checks.push_back(r.reason);
          return r;
        }
      }
      r.checks.push_back("simply_connected=" + std::string(sc ? "true" : "false") + " H₁ betti=" + std::to_string(hg.size() > 1 ? hg[1].betti : 0));

      r.reason = "consistent";
      return r;
    }

    // ── Differential geometry helpers ──────────────────────────────────

    virtual differential::OneForm metric() const
    {
      // Default flat metric: g = Σ dx_i ⊗ dx_i
      differential::OneForm g;
      g.dim = dimension();
      // Represented as 1-form for simplicity; actual metric is (0,2)-tensor
      return g;
    }

    virtual std::vector<std::vector<double>> riemann_curvature(const differential::Point& /*p*/) const
    {
      // Default flat: R=0
      int n = dimension();
      return std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    }

    virtual bool is_einstein() const { return false; }
    virtual bool is_kahler() const { return false; }
  };

  using AbstractVariety = AbstractManifold; // alias for backward compatibility

  // ── Sphere ──────────────────────────────────────────────────────────────

  struct SphereManifold : AbstractManifold
  {
    int n = 2;
    explicit SphereManifold(int dim = 2) : n(dim) {}
    std::string name() const override { return "S^" + std::to_string(n); }
    int dimension() const override { return n; }
    bool is_orientable() const override { return true; }
    bool is_compact() const override { return true; }
    bool is_simply_connected() const override { return n >= 2; }

    std::vector<homology::HomologyGroup> homology() const override
    {
      std::vector<homology::HomologyGroup> out(n + 1);
      for (int k = 0; k <= n; ++k) { if (k == 0 || k == n) out[k].betti = 1; }
      return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
      if (k < 0 || k > n) return homology::HomologyGroup{0, {}};
      homology::HomologyGroup g; g.betti = (k == 0 || k == n) ? 1 : 0; return g;
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
      if (k == n) return {1, {}, false};
      if (k < n) return {0, {}};
      return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
      homology::HomologyGroup g; g.betti = (k == 0 || k == n) ? 1 : 0; return g;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
      if (n == 0) return homology::SimplicialComplex{{{{0}}, {{1}}, {}, {}}};
      if (n == 1) return homology::circle_complex();
      if (n == 2) return homology::sphere_tetrahedron();
      return homology::make_simplex(n + 1);
    }
    int euler_characteristic() const override { return (n % 2 == 0) ? 2 : 0; }
    std::vector<std::vector<double>> riemann_curvature(const differential::Point& /*p*/) const override
    {
      // S^n has constant sectional curvature 1
      int d = n;
      std::vector<std::vector<double>> R(d, std::vector<double>(d, 0));
      for (int i = 0; i < d; ++i) R[i][i] = 1.0;
      return R;
    }
    bool is_einstein() const override { return true; }
  };
  using SphereVariety = SphereManifold;

  // ── Torus ───────────────────────────────────────────────────────────────

  struct TorusManifold : AbstractManifold
  {
    int dim = 2;
    explicit TorusManifold(int d = 2) : dim(d) {}
    std::string name() const override { return "T^" + std::to_string(dim); }
    int dimension() const override { return dim; }
    bool is_orientable() const override { return true; }
    bool is_compact() const override { return true; }
    std::vector<homology::HomologyGroup> homology() const override
    {
      std::vector<homology::HomologyGroup> out(dim + 1);
      for (int k = 0; k <= dim; ++k)
      {
        int num = 1, den = 1;
        for (int i = 0; i < k; ++i) { num *= (dim - i); den *= (k - i); }
        int b = (k == 0) ? 1 : num / den;
        if (dim == 2) { if (k == 0) b = 1; else if (k == 1) b = 2; else if (k == 2) b = 1; else b = 0; }
        out[k].betti = b;
      }
      return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
      auto h = homology(); if (k < 0 || k >= (int)h.size()) return {0, {}}; return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
      if (k == 1) return {dim, {}, false};
      return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override { return homology(k); }
    homology::SimplicialComplex to_simplicial() const override
    {
      if (dim == 1) return homology::circle_complex();
      if (dim == 2)
        return homology::SimplicialComplex{
            {{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}},
             {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {0, 3}, {1, 4}, {2, 5}},
             {{0, 1, 4}, {0, 4, 3}, {1, 2, 5}, {1, 5, 4}, {2, 0, 3}, {2, 3, 5}},
             {}}};
      return homology::make_simplex(dim + 1);
    }
    int euler_characteristic() const override { return 0; }
    bool is_einstein() const override { return true; } // flat torus
  };
  using TorusVariety = TorusManifold;

  // ── Projective ─────────────────────────────────────────────────────────

  struct ProjectiveManifold : AbstractManifold
  {
    std::string field = "R";
    int n = 2;
    ProjectiveManifold(std::string f = "R", int dim = 2) : field(std::move(f)), n(dim) {}
    std::string name() const override { return field + "P^" + std::to_string(n); }
    int dimension() const override { return n * (field == "C" ? 2 : 1); }
    bool is_orientable() const override { return field == "C" || n % 2 == 1; }
    bool is_compact() const override { return true; }
    std::vector<homology::HomologyGroup> homology() const override
    {
      std::vector<homology::HomologyGroup> out(dimension() + 1);
      if (field == "C") for (int k = 0; k <= n; ++k) out[2 * k].betti = 1;
      else
      {
        out[0].betti = 1;
        for (int k = 1; k < n; ++k) if (k % 2 == 1) out[k].torsion = {bigint(2)};
        if (n % 2 == 1) out[n].betti = 1; else if (n > 0) out[n].torsion = {bigint(2)};
      }
      return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
      auto h = homology(); if (k < 0 || k >= (int)h.size()) return {0, {}}; return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override
    {
      if (field == "C" && k == 2) return {1, {}, false};
      if (field == "R" && k == 1) return {1, {bigint(2)}, false};
      return {0, {}, true};
    }
    homology::HomologyGroup de_rham(int k) const override
    {
      homology::HomologyGroup g;
      if (field == "C") g.betti = (k % 2 == 0 && k <= 2 * n) ? 1 : 0;
      else g.betti = (k == 0 || (k == n && n % 2 == 1)) ? 1 : 0;
      return g;
    }
    homology::SimplicialComplex to_simplicial() const override
    {
      if (n == 1 && field == "R") return homology::circle_complex();
      return homology::make_simplex(n + 1);
    }
    int euler_characteristic() const override { return field == "C" ? n + 1 : (n % 2 == 0 ? 1 : 0); }
    bool is_kahler() const override { return field == "C"; }
  };
  using ProjectiveVariety = ProjectiveManifold;

  // ── Wedge / Product ────────────────────────────────────────────────────

  struct WedgeManifold : AbstractManifold
  {
    std::vector<std::unique_ptr<AbstractManifold>> parts;
    explicit WedgeManifold(std::vector<std::unique_ptr<AbstractManifold>> p) : parts(std::move(p)) {}
    std::string name() const override { return "Wedge"; }
    int dimension() const override
    {
      int d = 0; for (auto& pp : parts) d = std::max(d, pp->dimension()); return d;
    }
    std::vector<homology::HomologyGroup> homology() const override
    {
      int D = dimension(); std::vector<homology::HomologyGroup> out(D + 1); out[0].betti = 1;
      for (auto& pp : parts) { auto h = pp->homology(); for (int k = 1; k <= D && k < (int)h.size(); ++k) out[k].betti += h[k].betti; }
      return out;
    }
    homology::HomologyGroup homology(int k) const override
    {
      auto h = homology(); if (k < 0 || k >= (int)h.size()) return {0, {}}; return h[k];
    }
    homotopy::HomotopyGroup homotopy(int k) const override { return {0, {}, true}; }
    homology::HomologyGroup de_rham(int k) const override { return homology(k); }
    homology::SimplicialComplex to_simplicial() const override
    {
      if (!parts.empty()) return parts[0]->to_simplicial();
      return homology::SimplicialComplex{};
    }
    int euler_characteristic() const override { int e = 1; for (auto& pp : parts) e += pp->euler_characteristic() - 1; return e; }
  };
  using WedgeVariety = WedgeManifold;

  // ── Algebraic geometry helpers ─────────────────────────────────────────

  struct AffineScheme
  {
    // Ideal in k[x1..xn] given by polynomials as strings, e.g. {"x^2 + y^2 -1"}
    std::vector<std::string> equations;
    int ambient_dim = 0;
    std::string coordinate_ring() const
    {
      std::string s = "k[x1..x" + std::to_string(ambient_dim) + "]/(";
      for (size_t i = 0; i < equations.size(); ++i) { if (i) s += ", "; s += equations[i]; }
      s += ")";
      return s;
    }
    int krull_dimension() const
    {
      // For hypersurface in A^n, dim = n-1 (if irreducible)
      if (equations.empty()) return ambient_dim;
      return ambient_dim - (int)equations.size(); // naive
    }
    bool is_smooth() const
    {
      // Jacobian criterion: smooth if gradient not simultaneously zero on variety
      // For single equation f, check if f and ∂f/∂x_i have no common zero
      // Here we approximate: if equations are "x^2 + y^2 -1", then ∂f = (2x,2y) non-zero on circle
      return true; // placeholder for demonstration
    }
    bool is_irreducible() const { return true; }
    bool is_reduced() const { return true; }
    std::string to_string() const { return "Spec " + coordinate_ring(); }
  };

  struct Sheaf
  {
    std::string name;
    std::string type = "O_X"; // O_X, O_X(D), etc.
    bool is_coherent = true;
    bool is_locally_free = true;
    int rank = 1;
  };

  // ── Value helpers ──────────────────────────────────────────────────────

  NP_NODISCARD inline SphereManifold make_sphere(int n) { return SphereManifold(n); }
  NP_NODISCARD inline TorusManifold make_torus(int d = 2) { return TorusManifold(d); }
  NP_NODISCARD inline ProjectiveManifold make_real_projective(int n) { return ProjectiveManifold("R", n); }
  NP_NODISCARD inline ProjectiveManifold make_complex_projective(int n) { return ProjectiveManifold("C", n); }

  using AnyManifold = std::variant<SphereManifold, TorusManifold, ProjectiveManifold>;
  using AnyVariety = AnyManifold; // alias

  NP_NODISCARD inline std::vector<homology::HomologyGroup> homology(const AnyManifold& v) { return std::visit([](auto& x){ return x.homology(); }, v); }
  NP_NODISCARD inline homology::HomologyGroup homology(const AnyManifold& v, int k) { return std::visit([k](auto& x){ return x.homology(k); }, v); }
  NP_NODISCARD inline int euler_characteristic(const AnyManifold& v) { return std::visit([](auto& x){ return x.euler_characteristic(); }, v); }
  NP_NODISCARD inline std::string name(const AnyManifold& v) { return std::visit([](auto& x){ return x.name(); }, v); }

  NP_NODISCARD inline homology::SimplicialComplex sphere_complex(int n) { return SphereManifold(n).to_simplicial(); }
  NP_NODISCARD inline homology::SimplicialComplex torus_complex(int d = 2) { return TorusManifold(d).to_simplicial(); }

  NP_NODISCARD inline bool is_homotopy_equivalent(const AbstractManifold& A, const AbstractManifold& B)
  {
    return homotopy::is_homotopy_equivalent(A.to_simplicial(), B.to_simplicial()).equivalent;
  }

} // namespace np::manifold

// ── Backward compatibility: variety is now manifold ──────────────────────
namespace np::variety
{
  using AbstractVariety = manifold::AbstractManifold;
  using SphereVariety = manifold::SphereManifold;
  using TorusVariety = manifold::TorusManifold;
  using ProjectiveVariety = manifold::ProjectiveManifold;
  using WedgeVariety = manifold::WedgeManifold;
  using AnyVariety = manifold::AnyManifold;
  // Original factories return unique_ptr for backward compat
  inline auto sphere(int n) { return std::make_unique<manifold::SphereManifold>(n); }
  inline auto torus(int d = 2) { return std::make_unique<manifold::TorusManifold>(d); }
  inline auto projective_space(std::string f, int n) { return std::make_unique<manifold::ProjectiveManifold>(std::move(f), n); }
  inline auto real_projective(int n) { return std::make_unique<manifold::ProjectiveManifold>("R", n); }
  inline auto complex_projective(int n) { return std::make_unique<manifold::ProjectiveManifold>("C", n); }
  inline auto sphere_ptr(int n) { return sphere(n); }
  inline auto torus_ptr(int d = 2) { return torus(d); }
} // namespace np::variety

#endif // NP_MANIFOLD_HPP
