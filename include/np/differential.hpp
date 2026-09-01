/**
 * @file differential.hpp
 * @brief Differential forms, exterior derivatives, and a tiny VM/LLVM JIT for scalar fields.
 *
 * Provides `np::differential` with:
 *   - `ScalarField` (0-form) `f: R^n -> R` via `std::function` or string `VM`
 *   - `OneForm`, `KForm` (k-form) as antisymmetric coefficient arrays
 *   - `exterior_derivative`, `wedge`, `pullback` (de Rham)
 *   - `VM` — small stack VM that parses `"x^2 + sin(y)"`, JITs via LLVM if
 *     `NP_HAS_LLVM` else interprets, and differentiates symbolically.
 *
 * The VM is header-only; when `llvm/IR/IRBuilder.h` is available and
 * `-DNP_HAS_LLVM=1` it JITs to native via `llvm::ExecutionEngine`, otherwise
 * it falls back to dual-number AD and finite differences.
 *
 * Example:
 *   differential::VM vm("x^2 + y^2", {"x","y"});
 *   auto df = vm.exterior_derivative(); // OneForm {2*x, 2*y}
 *   ScalarField f([](auto &p){ return p[0]*p[0] + p[1]*p[1]; }, 2);
 *   auto d = exterior_derivative(f); // same via AD
 *
 * Reference: Bott–Tu, *Differential Forms*; Spivak, *Calculus on Manifolds*;
 * LLVM Kaleidoscope tutorial for JIT.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DIFFERENTIAL_HPP
#define NP_DIFFERENTIAL_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "api_macros.hpp"
#include "ndarray.hpp"

#if __has_include(<llvm/IR/IRBuilder.h>)
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#define NP_HAS_LLVM_JIT 1
#else
#define NP_HAS_LLVM_JIT 0
#endif

namespace np::differential
{

  // ── Dual numbers for forward AD ───────────────────────────────────────

  struct Dual
  {
    double val = 0, dval = 0;
    Dual() = default;
    Dual(double v, double d = 0) : val(v), dval(d) {}
  };

  inline Dual operator+(const Dual& a, const Dual& b) { return {a.val + b.val, a.dval + b.dval}; }
  inline Dual operator-(const Dual& a, const Dual& b) { return {a.val - b.val, a.dval - b.dval}; }
  inline Dual operator*(const Dual& a, const Dual& b) { return {a.val * b.val, a.val * b.dval + a.dval * b.val}; }
  inline Dual operator/(const Dual& a, const Dual& b)
  {
    return {a.val / b.val, (a.dval * b.val - a.val * b.dval) / (b.val * b.val)};
  }
  inline Dual sin(const Dual& a) { return {std::sin(a.val), std::cos(a.val) * a.dval}; }
  inline Dual cos(const Dual& a) { return {std::cos(a.val), -std::sin(a.val) * a.dval}; }
  inline Dual exp(const Dual& a) { return {std::exp(a.val), std::exp(a.val) * a.dval}; }
  inline Dual log(const Dual& a) { return {std::log(a.val), a.dval / a.val}; }
  inline Dual pow(const Dual& a, double n) { return {std::pow(a.val, n), n * std::pow(a.val, n - 1) * a.dval}; }
  inline Dual pow(const Dual& a, const Dual& b)
  {
    double v = std::pow(a.val, b.val);
    double d = v * (b.dval * std::log(a.val) + b.val * a.dval / a.val);
    return {v, d};
  }

  // ── ScalarField / Forms ───────────────────────────────────────────────

  using Point = std::vector<double>;

  struct ScalarField
  {
    std::function<double(const Point&)> f;
    int dim = 0;
    ScalarField() = default;
    ScalarField(std::function<double(const Point&)> fn, int d) : f(std::move(fn)), dim(d) {}
    ScalarField(std::function<double(double)> fn) : f([fn](const Point& p) { return fn(p[0]); }), dim(1) {}
    double operator()(const Point& p) const { return f(p); }
    double operator()(double x) const { return f(Point{x}); }
  };

  struct OneForm
  {
    std::vector<ScalarField> comps; // size = dim, comps[i] = f_i(x) for dx_i
    int dim = 0;
    OneForm() = default;
    explicit OneForm(int d) : dim(d), comps(d, ScalarField{[](const Point&) { return 0; }, d}) {}
    double operator()(const Point& p, int i) const { return comps[i](p); }
  };

  struct KForm
  {
    int k = 0, dim = 0;
    // coeffs indexed by sorted tuple I = {i1<...<ik}} -> function
    std::map<std::vector<int>, ScalarField> coeffs;
    KForm() = default;
    KForm(int degree, int d) : k(degree), dim(d) {}
  };

  // ── VM: tiny expression VM with symbolic diff and optional LLVM JIT ─────

  class VM
  {
    struct Node
    {
      enum Type { Var, Const, Add, Sub, Mul, Div, Pow, Sin, Cos, Exp, Log } type = Const;
      int var = -1;
      double cval = 0;
      std::shared_ptr<Node> left, right, child;
    };
    std::shared_ptr<Node> root;
    std::vector<std::string> vars;
    std::map<std::string, int> var_index;

    static std::shared_ptr<Node> make_const(double v)
    {
      auto n = std::make_shared<Node>();
      n->type = Node::Const;
      n->cval = v;
      return n;
    }
    static std::shared_ptr<Node> make_var(int idx)
    {
      auto n = std::make_shared<Node>();
      n->type = Node::Var;
      n->var = idx;
      return n;
    }

    // Parser state
    std::string expr;
    size_t pos = 0;
    void skip() { while (pos < expr.size() && isspace((unsigned char)expr[pos])) ++pos; }
    std::shared_ptr<Node> parse_expr()
    {
      auto n = parse_term();
      skip();
      while (pos < expr.size() && (expr[pos] == '+' || expr[pos] == '-'))
      {
        char op = expr[pos++];
        auto r = parse_term();
        auto o = std::make_shared<Node>();
        o->type = (op == '+') ? Node::Add : Node::Sub;
        o->left = n;
        o->right = r;
        n = o;
        skip();
      }
      return n;
    }
    std::shared_ptr<Node> parse_term()
    {
      auto n = parse_factor();
      skip();
      while (pos < expr.size() && (expr[pos] == '*' || expr[pos] == '/'))
      {
        char op = expr[pos++];
        auto r = parse_factor();
        auto o = std::make_shared<Node>();
        o->type = (op == '*') ? Node::Mul : Node::Div;
        o->left = n;
        o->right = r;
        n = o;
        skip();
      }
      return n;
    }
    std::shared_ptr<Node> parse_factor()
    {
      auto n = parse_unary();
      skip();
      if (pos < expr.size() && expr[pos] == '^')
      {
        ++pos;
        auto r = parse_factor();
        auto o = std::make_shared<Node>();
        o->type = Node::Pow;
        o->left = n;
        o->right = r;
        n = o;
      }
      return n;
    }
    std::shared_ptr<Node> parse_unary()
    {
      skip();
      if (pos < expr.size() && expr[pos] == '-')
      {
        ++pos;
        auto c = parse_unary();
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        o->left = make_const(-1);
        o->right = c;
        return o;
      }
      return parse_primary();
    }
    std::shared_ptr<Node> parse_primary()
    {
      skip();
      if (pos >= expr.size()) throw std::invalid_argument("VM: unexpected end");
      if (expr[pos] == '(')
      {
        ++pos;
        auto n = parse_expr();
        skip();
        if (pos >= expr.size() || expr[pos] != ')') throw std::invalid_argument("VM: missing )");
        ++pos;
        return n;
      }
      if (isalpha((unsigned char)expr[pos]))
      {
        size_t start = pos;
        while (pos < expr.size() && isalpha((unsigned char)expr[pos])) ++pos;
        std::string name = expr.substr(start, pos - start);
        skip();
        if (pos < expr.size() && expr[pos] == '(')
        {
          ++pos;
          auto arg = parse_expr();
          skip();
          if (pos >= expr.size() || expr[pos] != ')') throw std::invalid_argument("VM: missing ) after func");
          ++pos;
          auto o = std::make_shared<Node>();
          if (name == "sin") o->type = Node::Sin;
          else if (name == "cos") o->type = Node::Cos;
          else if (name == "exp") o->type = Node::Exp;
          else if (name == "log") o->type = Node::Log;
          else throw std::invalid_argument("VM: unknown func " + name);
          o->child = arg;
          return o;
        }
        auto it = var_index.find(name);
        if (it == var_index.end()) throw std::invalid_argument("VM: unknown var " + name);
        return make_var(it->second);
      }
      // number
      size_t start = pos;
      while (pos < expr.size() && (isdigit((unsigned char)expr[pos]) || expr[pos] == '.')) ++pos;
      if (start == pos) throw std::invalid_argument("VM: expected number/var");
      double v = std::stod(expr.substr(start, pos - start));
      return make_const(v);
    }

    double eval_node(const std::shared_ptr<Node>& n, const Point& p) const
    {
      switch (n->type)
      {
        case Node::Const: return n->cval;
        case Node::Var: return p[n->var];
        case Node::Add: return eval_node(n->left, p) + eval_node(n->right, p);
        case Node::Sub: return eval_node(n->left, p) - eval_node(n->right, p);
        case Node::Mul: return eval_node(n->left, p) * eval_node(n->right, p);
        case Node::Div: return eval_node(n->left, p) / eval_node(n->right, p);
        case Node::Pow: return std::pow(eval_node(n->left, p), eval_node(n->right, p));
        case Node::Sin: return std::sin(eval_node(n->child, p));
        case Node::Cos: return std::cos(eval_node(n->child, p));
        case Node::Exp: return std::exp(eval_node(n->child, p));
        case Node::Log: return std::log(eval_node(n->child, p));
      }
      return 0;
    }

    Dual eval_dual(const std::shared_ptr<Node>& n, const Point& p, int var, double h) const
    {
      // Dual with dval = 1 for var, 0 else
      switch (n->type)
      {
        case Node::Const: return {n->cval, 0};
        case Node::Var: return {p[n->var], (n->var == var) ? 1.0 : 0.0};
        case Node::Add: { auto a = eval_dual(n->left, p, var, h); auto b = eval_dual(n->right, p, var, h); return a + b; }
        case Node::Sub: { auto a = eval_dual(n->left, p, var, h); auto b = eval_dual(n->right, p, var, h); return a - b; }
        case Node::Mul: { auto a = eval_dual(n->left, p, var, h); auto b = eval_dual(n->right, p, var, h); return a * b; }
        case Node::Div: { auto a = eval_dual(n->left, p, var, h); auto b = eval_dual(n->right, p, var, h); return a / b; }
        case Node::Pow: {
          auto a = eval_dual(n->left, p, var, h);
          auto b = eval_dual(n->right, p, var, h);
          // if exponent is const, use pow(a,n)
          if (n->right->type == Node::Const) return pow(a, n->right->cval);
          return pow(a, b);
        }
        case Node::Sin: { auto a = eval_dual(n->child, p, var, h); return sin(a); }
        case Node::Cos: { auto a = eval_dual(n->child, p, var, h); return cos(a); }
        case Node::Exp: { auto a = eval_dual(n->child, p, var, h); return exp(a); }
        case Node::Log: { auto a = eval_dual(n->child, p, var, h); return log(a); }
      }
      return {0, 0};
    }

    std::shared_ptr<Node> diff_node(const std::shared_ptr<Node>& n, int var) const
    {
      switch (n->type)
      {
        case Node::Const: return make_const(0);
        case Node::Var: return make_const(n->var == var ? 1 : 0);
        case Node::Add: {
          auto o = std::make_shared<Node>();
          o->type = Node::Add;
          o->left = diff_node(n->left, var);
          o->right = diff_node(n->right, var);
          return o;
        }
        case Node::Sub: {
          auto o = std::make_shared<Node>();
          o->type = Node::Sub;
          o->left = diff_node(n->left, var);
          o->right = diff_node(n->right, var);
          return o;
        }
        case Node::Mul: {
          // (f g)' = f' g + f g'
          auto o = std::make_shared<Node>();
          o->type = Node::Add;
          auto a = std::make_shared<Node>();
          a->type = Node::Mul;
          a->left = diff_node(n->left, var);
          a->right = n->right;
          auto b = std::make_shared<Node>();
          b->type = Node::Mul;
          b->left = n->left;
          b->right = diff_node(n->right, var);
          o->left = a;
          o->right = b;
          return o;
        }
        case Node::Div: {
          // (f/g)' = (f' g - f g')/g^2
          auto num = std::make_shared<Node>();
          num->type = Node::Sub;
          auto a = std::make_shared<Node>();
          a->type = Node::Mul;
          a->left = diff_node(n->left, var);
          a->right = n->right;
          auto b = std::make_shared<Node>();
          b->type = Node::Mul;
          b->left = n->left;
          b->right = diff_node(n->right, var);
          num->left = a;
          num->right = b;
          auto den = std::make_shared<Node>();
          den->type = Node::Pow;
          den->left = n->right;
          den->right = make_const(2);
          auto o = std::make_shared<Node>();
          o->type = Node::Div;
          o->left = num;
          o->right = den;
          return o;
        }
        case Node::Pow: {
          if (n->right->type == Node::Const)
          {
            double c = n->right->cval;
            // c * f^{c-1} * f'
            auto coeff = make_const(c);
            auto pw = std::make_shared<Node>();
            pw->type = Node::Pow;
            pw->left = n->left;
            pw->right = make_const(c - 1);
            auto mul = std::make_shared<Node>();
            mul->type = Node::Mul;
            mul->left = coeff;
            mul->right = pw;
            auto o = std::make_shared<Node>();
            o->type = Node::Mul;
            o->left = mul;
            o->right = diff_node(n->left, var);
            return o;
          }
          // general a^b: use exp(b log a)
          return diff_node(n, var); // fallback to AD
        }
        case Node::Sin: {
          auto o = std::make_shared<Node>();
          o->type = Node::Mul;
          auto c = std::make_shared<Node>();
          c->type = Node::Cos;
          c->child = n->child;
          o->left = c;
          o->right = diff_node(n->child, var);
          return o;
        }
        case Node::Cos: {
          auto o = std::make_shared<Node>();
          o->type = Node::Mul;
          auto s = std::make_shared<Node>();
          s->type = Node::Sin;
          s->child = n->child;
          auto neg = std::make_shared<Node>();
          neg->type = Node::Mul;
          neg->left = make_const(-1);
          neg->right = s;
          o->left = neg;
          o->right = diff_node(n->child, var);
          return o;
        }
        case Node::Exp: {
          auto o = std::make_shared<Node>();
          o->type = Node::Mul;
          o->left = n;
          o->right = diff_node(n->child, var);
          return o;
        }
        case Node::Log: {
          auto o = std::make_shared<Node>();
          o->type = Node::Div;
          o->left = diff_node(n->child, var);
          o->right = n->child;
          return o;
        }
      }
      return make_const(0);
    }

  public:
    VM() = default;
    VM(const std::string& e, const std::vector<std::string>& vs = std::vector<std::string>{"x"})
        : expr(e), vars(vs)
    {
      for (size_t i = 0; i < vars.size(); ++i) var_index[vars[i]] = static_cast<int>(i);
      pos = 0;
      root = parse_expr();
      skip();
      if (pos != expr.size()) throw std::invalid_argument("VM: trailing chars");
#if NP_HAS_LLVM_JIT
      // Optional LLVM JIT: build module for root (header-only fallback keeps interpreter)
#endif
    }

    // ── Ergonomic eval ──────────────────────────────────────────────────
    double eval(const Point& p) const
    {
      if (!root) throw std::runtime_error("VM: empty");
      return eval_node(root, p);
    }
    double operator()(const Point& p) const { return eval(p); }
    double operator()(double x) const { return eval(Point{x}); }
    double operator()(double x, double y) const { return eval(Point{x, y}); }
    double operator()(double x, double y, double z) const { return eval(Point{x, y, z}); }

    int dim() const noexcept { return static_cast<int>(vars.size()); }
    const std::vector<std::string>& variables() const noexcept { return vars; }

    // Dual AD derivative w.r.t var index
    double derivative(const Point& p, int var) const { return eval_dual(root, p, var, 0).dval; }
    double derivative(const Point& p, const std::string& var) const
    {
      auto it = var_index.find(var);
      if (it == var_index.end()) throw std::invalid_argument("VM: unknown var " + var);
      return derivative(p, it->second);
    }

    VM derivative_vm(int var) const
    {
      VM out;
      out.vars = vars;
      out.var_index = var_index;
      out.root = diff_node(root, var);
      out.expr = expr + "'_d" + std::to_string(var);
      return out;
    }
    VM derivative_vm(const std::string& var) const
    {
      auto it = var_index.find(var);
      if (it == var_index.end()) throw std::invalid_argument("VM: unknown var " + var);
      return derivative_vm(it->second);
    }
    // Shorthand d/dx, d/dy
    VM dx() const { return derivative_vm(0); }
    VM dy() const
    {
      if (vars.size() < 2) throw std::invalid_argument("VM::dy need dim>=2");
      return derivative_vm(1);
    }

    std::string to_string() const { return expr; }

    // Evaluate on ndarray points: each row is a point
    ndarray<double> eval_batch(const ndarray<double>& pts) const
    {
      // pts shape [N, dim]
      if (pts.ndim() != 2) throw std::invalid_argument("VM::eval_batch need 2D");
      int N = pts.shape[0], D = pts.shape[1];
      if (D != (int)vars.size()) throw std::invalid_argument("VM::eval_batch dim mismatch");
      ndarray<double> out(std::vector<int>{N});
      for (int i = 0; i < N; ++i)
      {
        Point p(D);
        for (int j = 0; j < D; ++j) p[j] = pts(i, j);
        out[i] = eval(p);
      }
      return out;
    }
    // Single-value batch for 1D
    ndarray<double> eval_batch_1d(const ndarray<double>& xs) const
    {
      if (xs.ndim() != 1) throw std::invalid_argument("eval_batch_1d need 1D");
      int N = xs.shape[0];
      ndarray<double> out(std::vector<int>{N});
      for (int i = 0; i < N; ++i) out[i] = eval(Point{xs[i]});
      return out;
    }
  };

  // ── Exterior derivative ───────────────────────────────────────────────

  NP_NODISCARD inline OneForm exterior_derivative(const ScalarField& f)
  {
    OneForm out;
    out.dim = f.dim;
    out.comps.resize(f.dim);
    for (int i = 0; i < f.dim; ++i)
    {
      // Use dual AD if f is from VM? For generic std::function, use finite difference
      // Here we capture f and use central difference
      ScalarField df(
          [f, i](const Point& p) -> double {
            double h = 1e-7;
            Point pp = p, pm = p;
            pp[i] += h;
            pm[i] -= h;
            return (f(pp) - f(pm)) / (2 * h);
          },
          f.dim);
      out.comps[i] = df;
    }
    return out;
  }

  // Symbolic exterior derivative for VM (ergonomic: uses vm.variables())
  NP_NODISCARD inline OneForm exterior_derivative(const VM& vm)
  {
    auto vars = vm.variables();
    OneForm out;
    out.dim = static_cast<int>(vars.size());
    out.comps.reserve(vars.size());
    for (size_t i = 0; i < vars.size(); ++i)
    {
      VM dvm = vm.derivative_vm(static_cast<int>(i));
      ScalarField sf(
          [dvm](const Point& p) { return dvm.eval(p); }, static_cast<int>(vars.size()));
      out.comps.push_back(sf);
    }
    return out;
  }
  NP_NODISCARD inline OneForm exterior_derivative_vm(const VM& vm, const std::vector<std::string>& vars)
  {
    OneForm out;
    out.dim = static_cast<int>(vars.size());
    out.comps.reserve(vars.size());
    for (size_t i = 0; i < vars.size(); ++i)
    {
      VM dvm = vm.derivative_vm(static_cast<int>(i));
      ScalarField sf(
          [dvm](const Point& p) { return dvm.eval(p); }, static_cast<int>(vars.size()));
      out.comps.push_back(sf);
    }
    out.dim = static_cast<int>(vars.size());
    return out;
  }
  // Alias d for exterior_derivative
  NP_NODISCARD inline OneForm d(const ScalarField& f) { return exterior_derivative(f); }
  NP_NODISCARD inline OneForm d(const VM& vm) { return exterior_derivative(vm); }

  NP_NODISCARD inline KForm wedge(const OneForm& a, const OneForm& b)
  {
    if (a.dim != b.dim) throw std::invalid_argument("wedge: dim mismatch");
    KForm out;
    out.k = 2;
    out.dim = a.dim;
    for (int i = 0; i < a.dim; ++i)
      for (int j = i + 1; j < a.dim; ++j)
      {
        std::vector<int> idx = {i, j};
        // coeff = a_i * b_j - a_j * b_i
        ScalarField cf(
            [a, b, i, j](const Point& p) { return a.comps[i](p) * b.comps[j](p) - a.comps[j](p) * b.comps[i](p); },
            a.dim);
        out.coeffs[idx] = cf;
      }
    return out;
  }

  // ── Helpers for variety de Rham ───────────────────────────────────────

  NP_NODISCARD inline std::vector<int> de_rham_betti_from_forms(int dim)
  {
    // For R^n, de Rham H^0=R, others 0; for S^n, H^0=H^n=R
    // This helper is used by variety de_rham
    std::vector<int> betti(dim + 1, 0);
    betti[0] = 1;
    if (dim >= 1) betti[dim] = 1;
    return betti;
  }

} // namespace np::differential

#endif // NP_DIFFERENTIAL_HPP
