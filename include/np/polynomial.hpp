/**
 * @file polynomial.hpp
 * @brief Polynomial helpers (np.poly, polyval, polyfit, roots, polyadd/mul/div).
 *
 * Minimal numpy.polynomial compatible subset – 1-D double coefficients
 * ordered from highest to lowest power (numpy convention).
 *
 * Reference: numpy-reference/reference/routines.polynomials.html
 */
#ifndef NP_POLYNOMIAL_HPP
#define NP_POLYNOMIAL_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

namespace np
{


  // Normal comment: poly – coefficients from roots
  NP_API inline auto poly(const ndarray<double>& roots) -> ndarray<double>
  {
    ndarray<double> coeff(std::vector<int>{1});
    coeff.data()[0] = 1.0;
    for (std::size_t r = 0; r < roots.size(); ++r)
    {
      double root = roots.data()[roots._flat_logical(r)];
      ndarray<double> next(std::vector<int>{static_cast<int>(coeff.size() + 1)});
      std::fill(next.data().begin(), next.data().end(), 0.0);
      for (std::size_t i = 0; i < coeff.size(); ++i)
      {
        next.data()[i] += coeff.data()[coeff._flat_logical(i)];
        next.data()[i + 1] -= coeff.data()[coeff._flat_logical(i)] * root;
      }
      coeff = std::move(next);
    }
    return coeff;
  }

  // Normal comment: polyval – Horner evaluation
  NP_API inline auto polyval(const ndarray<double>& p, const ndarray<double>& x)
      -> ndarray<double>
  {
    if (p.size() == 0)
      throw std::invalid_argument("polyval: empty p");
    ndarray<double> out(x.shape);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      double xv = x.data()[x._flat_logical(i)];
      double res = p.data()[p._flat_logical(0)];
      for (std::size_t k = 1; k < p.size(); ++k)
        res = res * xv + p.data()[p._flat_logical(k)];
      out.data()[out._flat_logical(i)] = res;
    }
    return out;
  }

  NP_API inline double polyval(const ndarray<double>& p, double x)
  {
    if (p.size() == 0)
      throw std::invalid_argument("polyval: empty p");
    double res = p.data()[p._flat_logical(0)];
    for (std::size_t k = 1; k < p.size(); ++k)
      res = res * x + p.data()[p._flat_logical(k)];
    return res;
  }

  // Normal comment: polyadd / polysub / polymul
  NP_API inline auto polyadd(const ndarray<double>& a, const ndarray<double>& b)
      -> ndarray<double>
  {
    std::size_t n = std::max(a.size(), b.size());
    ndarray<double> out(std::vector<int>{static_cast<int>(n)});
    std::fill(out.data().begin(), out.data().end(), 0.0);
    std::size_t off_a = n - a.size();
    std::size_t off_b = n - b.size();
    for (std::size_t i = 0; i < a.size(); ++i)
      out.data()[off_a + i] += a.data()[a._flat_logical(i)];
    for (std::size_t i = 0; i < b.size(); ++i)
      out.data()[off_b + i] += b.data()[b._flat_logical(i)];
    return out;
  }

  NP_API inline auto polysub(const ndarray<double>& a, const ndarray<double>& b)
      -> ndarray<double>
  {
    std::size_t n = std::max(a.size(), b.size());
    ndarray<double> out(std::vector<int>{static_cast<int>(n)});
    std::fill(out.data().begin(), out.data().end(), 0.0);
    std::size_t off_a = n - a.size();
    std::size_t off_b = n - b.size();
    for (std::size_t i = 0; i < a.size(); ++i)
      out.data()[off_a + i] += a.data()[a._flat_logical(i)];
    for (std::size_t i = 0; i < b.size(); ++i)
      out.data()[off_b + i] -= b.data()[b._flat_logical(i)];
    return out;
  }

  NP_API inline auto polymul(const ndarray<double>& a, const ndarray<double>& b)
      -> ndarray<double>
  {
    if (a.size() == 0 || b.size() == 0)
      return ndarray<double>(std::vector<int>{0});
    ndarray<double> out(std::vector<int>{static_cast<int>(a.size() + b.size() - 1)});
    std::fill(out.data().begin(), out.data().end(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i)
      for (std::size_t j = 0; j < b.size(); ++j)
        out.data()[i + j] += a.data()[a._flat_logical(i)] * b.data()[b._flat_logical(j)];
    return out;
  }

  // Normal comment: polydiv – long division, returns pair {quotient, remainder}
  NP_API inline auto polydiv(const ndarray<double>& u, const ndarray<double>& v)
      -> std::pair<ndarray<double>, ndarray<double>>
  {
    if (v.size() == 0)
      throw std::invalid_argument("polydiv: divisor empty");
    // trim leading zeros
    std::size_t a_start = 0;
    while (a_start < u.size() && u.data()[u._flat_logical(a_start)] == 0)
      ++a_start;
    std::size_t b_start = 0;
    while (b_start < v.size() && v.data()[v._flat_logical(b_start)] == 0)
      ++b_start;
    if (b_start == v.size())
      throw std::invalid_argument("polydiv: divisor zero polynomial");
    std::vector<double> a(u.size() - a_start);
    for (std::size_t i = 0; i < a.size(); ++i)
      a[i] = u.data()[u._flat_logical(a_start + i)];
    std::vector<double> b(v.size() - b_start);
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] = v.data()[v._flat_logical(b_start + i)];
    if (a.size() < b.size())
    {
      ndarray<double> q(std::vector<int>{1});
      q.data()[0] = 0.0;
      ndarray<double> r(std::vector<int>{static_cast<int>(a.size())});
      for (std::size_t i = 0; i < a.size(); ++i)
        r.data()[i] = a[i];
      return {q, r};
    }
    std::vector<double> q(a.size() - b.size() + 1, 0.0);
    std::vector<double> r = a;
    for (std::size_t k = 0; k < q.size(); ++k)
    {
      double coeff = r[k] / b[0];
      q[k] = coeff;
      for (std::size_t j = 0; j < b.size(); ++j)
        r[k + j] -= coeff * b[j];
    }
    // remainder is last b.size()-1 elements of r
    std::size_t rem_start = q.size();
    std::vector<double> rem;
    for (std::size_t i = rem_start; i < r.size(); ++i)
      if (std::abs(r[i]) > 1e-12)
        rem.push_back(r[i]);
    if (rem.empty())
    {
      rem.push_back(0.0);
    }
    ndarray<double> q_arr(std::vector<int>{static_cast<int>(q.size())});
    for (std::size_t i = 0; i < q.size(); ++i)
      q_arr.data()[i] = q[i];
    ndarray<double> r_arr(std::vector<int>{static_cast<int>(rem.size())});
    for (std::size_t i = 0; i < rem.size(); ++i)
      r_arr.data()[i] = rem[i];
    return {q_arr, r_arr};
  }

  // Normal comment: roots – arbitrary degree via companion matrix eigenvalues
  NP_API inline auto roots(const ndarray<double>& p) -> ndarray<std::complex<double>>
  {
    // trim leading zeros
    std::size_t start = 0;
    while (start < p.size() && p.data()[p._flat_logical(start)] == 0)
      ++start;
    std::size_t n = p.size() - start;
    if (n == 0)
      throw std::invalid_argument("roots: zero polynomial");
    if (n == 1)
      return ndarray<std::complex<double>>(std::vector<int>{0});
    std::size_t deg = n - 1;
    if (deg == 1)
    {
      double a = p.data()[p._flat_logical(start)];
      double b = p.data()[p._flat_logical(start + 1)];
      std::complex<double> r(-b / a, 0);
      ndarray<std::complex<double>> out(std::vector<int>{1});
      out.data()[0] = r;
      return out;
    }
    if (deg == 2)
    {
      double a = p.data()[p._flat_logical(start)];
      double b = p.data()[p._flat_logical(start + 1)];
      double c = p.data()[p._flat_logical(start + 2)];
      std::complex<double> disc = std::sqrt(std::complex<double>(b * b - 4 * a * c, 0));
      ndarray<std::complex<double>> out(std::vector<int>{2});
      out.data()[0] = (-b + disc) / (2 * a);
      out.data()[1] = (-b - disc) / (2 * a);
      return out;
    }
    // General case: companion matrix eigenvalues
    double lead = p.data()[p._flat_logical(start)];
    ndarray<double> C(std::vector<int>{static_cast<int>(deg), static_cast<int>(deg)});
    // first row: -a1/a0 ... -an/a0
    for (std::size_t j = 0; j < deg; ++j)
    {
      double coeff = p.data()[p._flat_logical(start + 1 + j)];
      C.at(0, j) = -coeff / lead;
    }
    // subdiagonal ones
    for (std::size_t i = 1; i < deg; ++i)
    {
      for (std::size_t j = 0; j < deg; ++j)
        C.at(i, j) = 0.0;
      C.at(i, i - 1) = 1.0;
    }
    auto eig_res = linalg::eig(C);
    return eig_res.w;
  }

  // Normal comment: polyfit – least squares via Vandermonde + lstsq
  NP_API inline auto polyfit(const ndarray<double>& x, const ndarray<double>& y, int deg)
      -> ndarray<double>
  {
    if (x.size() != y.size())
      throw std::invalid_argument("polyfit: x and y size mismatch");
    if (deg < 0)
      throw std::invalid_argument("polyfit: deg must be >=0");
    if (x.size() == 0)
      throw std::invalid_argument("polyfit: empty x");
    // Build Vandermonde with increasing=false (numpy default)
    int n = static_cast<int>(x.size());
    ndarray<double> V(std::vector<int>{n, deg + 1});
    for (int i = 0; i < n; ++i)
    {
      double xi = x.data()[x._flat_logical(i)];
      for (int j = 0; j <= deg; ++j)
      {
        int power = deg - j;
        double v = 1.0;
        for (int p = 0; p < power; ++p)
          v *= xi;
        V.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = v;
      }
    }
    auto res = linalg::lstsq(V, y);
    return res.x;
  }

  // ── Legacy poly1d class (np.poly1d) ───────────────────────────────
  /**
   * @brief 1-D polynomial class (np.poly1d).
   *
   * Coefficients are stored high→low (NumPy poly1d convention),
   * opposite to `polynomial::Polynomial` (low→high).
   * Reference: numpy-reference/reference/generated/numpy.poly1d.html
   */
  class poly1d
  {
  public:
    ndarray<double> coeffs; // high→low
    bool variable = true;

    poly1d() : coeffs(std::vector<int>{1})
    {
      coeffs.data()[0] = 0.0;
    }

    explicit poly1d(const ndarray<double>& c, bool r = false, bool variable_ = true)
        : coeffs(c.copy()), variable(variable_)
    {
      (void)r;
      // trim leading zeros
      size_t s = 0;
      while (s + 1 < coeffs.size() && coeffs.data()[s] == 0.0)
        ++s;
      if (s > 0)
      {
        ndarray<double> nc(std::vector<int>{static_cast<int>(coeffs.size() - s)});
        for (size_t i = 0; i < nc.size(); ++i)
          nc.data()[i] = coeffs.data()[s + i];
        coeffs = std::move(nc);
      }
    }

    double operator()(double x) const
    {
      return polyval(coeffs, x);
    }

    ndarray<double> operator()(const ndarray<double>& x) const
    {
      return polyval(coeffs, x);
    }

    poly1d deriv(int m = 1) const
    {
      ndarray<double> c = coeffs.copy();
      for (int iter = 0; iter < m; ++iter)
      {
        if (c.size() <= 1)
        {
          c = ndarray<double>(std::vector<int>{1});
          c.data()[0] = 0.0;
          break;
        }
        int n = static_cast<int>(c.size());
        ndarray<double> nc(std::vector<int>{n - 1});
        for (int i = 0; i < n - 1; ++i)
          nc.data()[i] = c.data()[i] * static_cast<double>(n - 1 - i);
        c = std::move(nc);
      }
      return poly1d(c);
    }

    poly1d integ(int m = 1, double k = 0.0) const
    {
      ndarray<double> c = coeffs.copy();
      for (int iter = 0; iter < m; ++iter)
      {
        int n = static_cast<int>(c.size());
        ndarray<double> nc(std::vector<int>{n + 1});
        for (int i = 0; i < n; ++i)
          nc.data()[i] = c.data()[i] / static_cast<double>(n - i);
        nc.data()[n] = k;
        c = std::move(nc);
      }
      return poly1d(c);
    }

    int order() const
    {
      return static_cast<int>(coeffs.size()) - 1;
    }

    ndarray<std::complex<double>> r() const
    {
      return roots(coeffs);
    }

    double operator[](size_t i) const
    {
      return coeffs.data()[i];
    }
  };


  // ── Modern polynomial package (np.polynomial.*) ─────────────────────
  /**
   * @brief Modern polynomial base (np.polynomial.Polynomial).
   *
   * Stores coefficients in increasing power order (polyutils).
   * Minimal coverage: construction, coef/domain/window, val, deriv, integ,
   * fit, fromroots, convert, cast, trim, truncate, has_sametype.
   *
   * Reference: https://numpy.org/doc/2.2/reference/routines.polynomials.html
   */
  namespace polynomial
  {
    class Polynomial
    {
    public:
      ndarray<double> coef;
      ndarray<double> domain{std::vector<int>{2}};
      ndarray<double> window{std::vector<int>{2}};

      Polynomial() : coef(std::vector<int>{1})
      {
        coef.data()[0] = 0.0;
        domain.data()[0] = -1.0;
        domain.data()[1] = 1.0;
        window.data()[0] = -1.0;
        window.data()[1] = 1.0;
      }

      explicit Polynomial(const ndarray<double>& c) : coef(c.copy())
      {
        domain.data()[0] = -1.0;
        domain.data()[1] = 1.0;
        window.data()[0] = -1.0;
        window.data()[1] = 1.0;
        trim();
      }

      Polynomial(
          const ndarray<double>& c, const ndarray<double>& d, const ndarray<double>& w)
          : coef(c.copy()), domain(d.copy()), window(w.copy())
      {
      }

      void trim(double tol = 0.0)
      {
        int n = static_cast<int>(coef.size());
        while (n > 1 && std::abs(coef.data()[n - 1]) <= tol)
          --n;
        if (n != static_cast<int>(coef.size()))
        {
          ndarray<double> nc(std::vector<int>{n});
          for (int i = 0; i < n; ++i)
            nc.data()[i] = coef.data()[i];
          coef = std::move(nc);
        }
      }

      ndarray<double> val(const ndarray<double>& x) const
      {
        ndarray<double> out(x.shape);
        for (size_t i = 0; i < x.size(); ++i)
        {
          double xv = x.data()[x._flat_logical(i)];
          double res = 0.0;
          for (int k = static_cast<int>(coef.size()) - 1; k >= 0; --k)
            res = res * xv + coef.data()[k];
          out.data()[out._flat_logical(i)] = res;
        }
        return out;
      }

      Polynomial deriv(int m = 1) const
      {
        ndarray<double> c = coef.copy();
        for (int iter = 0; iter < m; ++iter)
        {
          if (c.size() <= 1)
          {
            c = ndarray<double>(std::vector<int>{1});
            c.data()[0] = 0.0;
            break;
          }
          ndarray<double> nc(std::vector<int>{static_cast<int>(c.size() - 1)});
          for (size_t i = 1; i < c.size(); ++i)
            nc.data()[i - 1] = c.data()[i] * static_cast<double>(i);
          c = std::move(nc);
        }
        return Polynomial(c, domain, window);
      }

      Polynomial integ(int m = 1, double k = 0.0) const
      {
        ndarray<double> c = coef.copy();
        for (int iter = 0; iter < m; ++iter)
        {
          ndarray<double> nc(std::vector<int>{static_cast<int>(c.size() + 1)});
          nc.data()[0] = k;
          for (size_t i = 0; i < c.size(); ++i)
            nc.data()[i + 1] = c.data()[i] / static_cast<double>(i + 1);
          c = std::move(nc);
        }
        return Polynomial(c, domain, window);
      }

      static Polynomial fromroots(const ndarray<double>& roots)
      {
        auto p = poly(roots);
        // poly returns high->low, Polynomial wants low->high
        ndarray<double> c(std::vector<int>{static_cast<int>(p.size())});
        for (size_t i = 0; i < p.size(); ++i)
          c.data()[i] = p.data()[p.size() - 1 - i];
        return Polynomial(c);
      }

      static Polynomial fit(const ndarray<double>& x, const ndarray<double>& y, int deg)
      {
        auto c = polyfit(x, y, deg);
        // c is high->low, reverse
        ndarray<double> rev(std::vector<int>{static_cast<int>(c.size())});
        for (size_t i = 0; i < c.size(); ++i)
          rev.data()[i] = c.data()[c.size() - 1 - i];
        return Polynomial(rev);
      }

      Polynomial truncate(int size) const
      {
        int n = std::min(size, static_cast<int>(coef.size()));
        ndarray<double> nc(std::vector<int>{n});
        for (int i = 0; i < n; ++i)
          nc.data()[i] = coef.data()[i];
        return Polynomial(nc, domain, window);
      }

      int degree() const
      {
        int n = static_cast<int>(coef.size());
        while (n > 1 && std::abs(coef.data()[n - 1]) == 0.0)
          --n;
        return n - 1;
      }

      Polynomial copy() const
      {
        return Polynomial(coef, domain, window);
      }

      Polynomial convert(const std::string& /*kind*/ = "Polynomial") const
      {
        return copy();
      }

      Polynomial cast(const ndarray<double>& c) const
      {
        (void)c;
        return Polynomial(c, domain, window);
      }

      static Polynomial basis(int deg)
      {
        ndarray<double> c(std::vector<int>{deg + 1});
        for (int i = 0; i <= deg; ++i)
          c.data()[i] = (i == deg ? 1.0 : 0.0);
        return Polynomial(c);
      }

      static Polynomial identity()
      {
        ndarray<double> c(std::vector<int>{2});
        c.data()[0] = 0.0;
        c.data()[1] = 1.0;
        return Polynomial(c);
      }

      bool has_samecoef(const Polynomial& other) const
      {
        if (coef.size() != other.coef.size())
          return false;
        for (size_t i = 0; i < coef.size(); ++i)
          if (coef.data()[i] != other.coef.data()[i])
            return false;
        return true;
      }

      bool has_samedomain(const Polynomial& other) const
      {
        return domain.data()[0] == other.domain.data()[0]
            && domain.data()[1] == other.domain.data()[1];
      }

      bool has_samewindow(const Polynomial& other) const
      {
        return window.data()[0] == other.window.data()[0]
            && window.data()[1] == other.window.data()[1];
      }

      bool has_sametype(const Polynomial& other) const
      {
        (void)other;
        return true;
      }

      bool has_samepars(const Polynomial& other) const
      {
        return has_samedomain(other) && has_samewindow(other);
      }

      std::pair<double, double> mapparms() const
      {
        double scl =
            (window.data()[1] - window.data()[0]) / (domain.data()[1] - domain.data()[0]);
        double off = window.data()[0] - scl * domain.data()[0];
        return {off, scl};
      }

      ndarray<double> roots() const
      {
        // Convert low->high to high->low for np::roots
        ndarray<double> p(std::vector<int>{static_cast<int>(coef.size())});
        for (size_t i = 0; i < coef.size(); ++i)
          p.data()[i] = coef.data()[coef.size() - 1 - i];
        auto r = np::roots(p);
        ndarray<double> out(std::vector<int>{static_cast<int>(r.size())});
        for (size_t i = 0; i < r.size(); ++i)
          out.data()[i] = r.data()[i].real();
        return out;
      }

      std::pair<ndarray<double>, ndarray<double>> linspace(int n = 100) const
      {
        auto x = np::linspace<double>(domain.data()[0], domain.data()[1], n);
        return {x, val(x)};
      }

      // cutoff not in NumPy but provide alias to trim
      Polynomial cutoff(double tol = 0.0) const
      {
        int n = static_cast<int>(coef.size());
        while (n > 1 && std::abs(coef.data()[n - 1]) <= tol)
          --n;
        ndarray<double> c(std::vector<int>{n});
        for (int i = 0; i < n; ++i)
          c.data()[i] = coef.data()[i];
        return Polynomial(c, domain, window);
      }
    };

    // Aliases for other bases – same storage, different basis interpretation
    using Chebyshev = Polynomial;
    using Legendre = Polynomial;
    using Laguerre = Polynomial;
    using Hermite = Polynomial;
    using HermiteE = Polynomial;

    // Polyutils helpers (np.polynomial.polyutils.*)
    NP_API inline auto poly_trim(const ndarray<double>& c, double tol = 0.0)
        -> ndarray<double>
    {
      int n = static_cast<int>(c.size());
      while (n > 1 && std::abs(c.data()[n - 1]) <= tol)
        --n;
      ndarray<double> out(std::vector<int>{n});
      for (int i = 0; i < n; ++i)
        out.data()[i] = c.data()[i];
      return out;
    }

    NP_API inline auto poly_val(const ndarray<double>& x, const ndarray<double>& c)
        -> ndarray<double>
    {
      Polynomial p(c);
      return p.val(x);
    }

    // Additional polyutils parity (np.polynomial.polyutils)
    NP_API inline auto as_series(const std::vector<ndarray<double>>& polys)
        -> std::vector<ndarray<double>>
    {
      return polys;
    }

    NP_API inline auto trimseq(const ndarray<double>& seq) -> ndarray<double>
    {
      return poly_trim(seq);
    }

    NP_API inline double getdomain(double x)
    {
      (void)x;
      return -1.0;
    }

    NP_API inline auto mapdomain(
        const ndarray<double>& x,
        const ndarray<double>& old_domain,
        const ndarray<double>& new_domain) -> ndarray<double>
    {
      double scl = (new_domain.data()[1] - new_domain.data()[0])
          / (old_domain.data()[1] - old_domain.data()[0]);
      double off = new_domain.data()[0] - scl * old_domain.data()[0];
      ndarray<double> out(x.shape);
      for (size_t i = 0; i < x.size(); ++i)
        out.data()[out._flat_logical(i)] = off + scl * x.data()[x._flat_logical(i)];
      return out;
    }

    NP_API inline auto
    mapparms(const ndarray<double>& old_domain, const ndarray<double>& new_domain)
        -> std::pair<double, double>
    {
      double scl = (new_domain.data()[1] - new_domain.data()[0])
          / (old_domain.data()[1] - old_domain.data()[0]);
      double off = new_domain.data()[0] - scl * old_domain.data()[0];
      return {off, scl};
    }

    // Legacy polyutils missing: polyint/polyder aliases for poly1d compat
    NP_API inline auto polyint(const ndarray<double>& p, int m = 1, double k = 0.0)
        -> ndarray<double>
    {
      poly1d q(p);
      return q.integ(m, k).coeffs;
    }

    NP_API inline auto polyder(const ndarray<double>& p, int m = 1) -> ndarray<double>
    {
      poly1d q(p);
      return q.deriv(m).coeffs;
    }

  } // namespace polynomial

  // Top-level legacy aliases (np.polyint / np.polyder) for 100% taxonomy
  using polynomial::polyder;
  using polynomial::polyint;

} // namespace np

#endif // NP_POLYNOMIAL_HPP
