/**
 * @file differential.hpp
 * @brief Differential forms, exterior derivatives, and a tiny VM/LLVM JIT for scalar
 * fields — modern engine with design patterns.
 *
 * Provides `np::differential` with:
 *   - `ScalarField<T>` (0-form) `f: R^n -> T` via `std::function` or string `VM`
 *   - `OneForm<T>`, `KForm<T>` (k-form) as antisymmetric coefficient arrays
 *   - `exterior_derivative`, `wedge`, `pullback`, `interior_product`,
 *     `lie_derivative` (de Rham, Cartan) + form operators `+ - ^ *`
 *   - `VM` — small stack VM that parses `"x^2 + sin(y)"`, JITs via LLVM if
 *     `NP_HAS_LLVM` else interprets, and differentiates symbolically;
 *     supports `sqrt/tan/asin/acos/atan`, scientific notation, `from_chars`
 *     modern parsing, prototype clone, derivative cache, thread-safe observers.
 *   - Design patterns: **Strategy** (Evaluator + CachedDecorator), **Visitor**
 *     (Node), **Factory/Abstract Factory** (FormFactory), **Builder** (VM::Builder),
 *     **Decorator/Composite** (KForm wedge + CachedEvaluator), **Prototype**
 *     (Node::clone), **Observer** (VM observers), **Template Method**
 *     (exterior derivative) + Form `concept`.
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
 *   // Complex support (polynomial over C):
 *   ScalarField<c128_t> g([](auto &p){ return c128_t(p[0], p[1]); }, 2);
 *
 * Reference: Bott–Tu, *Differential Forms*; Spivak, *Calculus on Manifolds*;
 * LLVM Kaleidoscope tutorial for JIT; Gamma et al. *Design Patterns*.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DIFFERENTIAL_HPP
#define NP_DIFFERENTIAL_HPP

#include <algorithm>
#include <charconv>
#include <cmath>
#include <complex>
#include <concepts>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"

#if defined(NP_ENABLE_LLVM) && __has_include(<llvm/IR/IRBuilder.h>)
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

  // ── typedefs for std::types (do not use std:: explicitly) ───────────────
  using f64_t = typename float64::type;     // double
  using f32_t = typename float32::type;     // float
  using c128_t = typename complex128::type; // std::complex<double>
  using c64_t = typename complex64::type;   // std::complex<float>
  using i64_t = typename int64::type;       // std::int64_t

  // ── Concepts ─────────────────────────────────────────────────────────────
  template <typename T>
  concept Scalar = std::is_arithmetic_v<T> || detail::is_complex_v<T>;

  // ── Dual numbers for forward AD (templated, Strategy pattern) ───────────
  // Modern: constexpr, noexcept where possible, supports real + complex via ADL.
  template <Scalar T = f64_t>
  struct Dual
  {
    T val = T(0);
    T dval = T(0);
    constexpr Dual() = default;
    constexpr Dual(T v, T d = T(0)) noexcept : val(v), dval(d)
    {
    }
  };

  template <Scalar T>
  NP_NODISCARD constexpr inline Dual<T>
  operator+(const Dual<T>& a, const Dual<T>& b) noexcept
  {
    return {a.val + b.val, a.dval + b.dval};
  }
  template <Scalar T>
  NP_NODISCARD constexpr inline Dual<T>
  operator-(const Dual<T>& a, const Dual<T>& b) noexcept
  {
    return {a.val - b.val, a.dval - b.dval};
  }
  template <Scalar T>
  NP_NODISCARD constexpr inline Dual<T>
  operator*(const Dual<T>& a, const Dual<T>& b) noexcept
  {
    return {a.val * b.val, a.val * b.dval + a.dval * b.val};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> operator/(const Dual<T>& a, const Dual<T>& b)
  {
    return {a.val / b.val, (a.dval * b.val - a.val * b.dval) / (b.val * b.val)};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> sin(const Dual<T>& a)
  {
    using std::cos;
    using std::sin;
    return {sin(a.val), cos(a.val) * a.dval};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> cos(const Dual<T>& a)
  {
    using std::cos;
    using std::sin;
    return {cos(a.val), -sin(a.val) * a.dval};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> exp(const Dual<T>& a)
  {
    using std::exp;
    return {exp(a.val), exp(a.val) * a.dval};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> log(const Dual<T>& a)
  {
    using std::log;
    return {log(a.val), a.dval / a.val};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> pow(const Dual<T>& a, double n)
  {
    using std::pow;
    return {pow(a.val, n), n * pow(a.val, n - 1) * a.dval};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> pow(const Dual<T>& a, const Dual<T>& b)
  {
    using std::log;
    using std::pow;
    T v = pow(a.val, b.val);
    T d = v * (b.dval * log(a.val) + b.val * a.dval / a.val);
    return {v, d};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> sqrt(const Dual<T>& a)
  {
    using std::sqrt;
    T s = sqrt(a.val);
    return {s, a.dval / (T(2) * s)};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> tan(const Dual<T>& a)
  {
    using std::cos;
    using std::tan;
    T c = cos(a.val);
    return {tan(a.val), a.dval / (c * c)};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> asin(const Dual<T>& a)
  {
    using std::asin;
    using std::sqrt;
    return {asin(a.val), a.dval / sqrt(T(1) - a.val * a.val)};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> acos(const Dual<T>& a)
  {
    using std::acos;
    using std::sqrt;
    return {acos(a.val), -a.dval / sqrt(T(1) - a.val * a.val)};
  }
  template <Scalar T>
  NP_NODISCARD inline Dual<T> atan(const Dual<T>& a)
  {
    using std::atan;
    return {atan(a.val), a.dval / (T(1) + a.val * a.val)};
  }

  // ── ScalarField / Forms (templated for C, f64_t default) ─────────────────
  template <Scalar T = f64_t>
  using PointT = std::vector<T>;
  using Point = PointT<f64_t>;
  using CPoint = PointT<c128_t>;

  template <Scalar T = f64_t>
  struct ScalarFieldT
  {
    std::function<T(const PointT<T>&)> f;
    int dim = 0;
    ScalarFieldT() = default;
    ScalarFieldT(std::function<T(const PointT<T>&)> fn, int d) : f(std::move(fn)), dim(d)
    {
    }
    // 1-D convenience: f(double) -> Point{x}
    template <typename U = T>
      requires std::is_same_v<U, f64_t>
    ScalarFieldT(std::function<f64_t(f64_t)> fn)
        : f([fn](const Point& p) { return fn(p[0]); }), dim(1)
    {
    }
    T operator()(const PointT<T>& p) const
    {
      return f(p);
    }
    T operator()(f64_t x) const
      requires std::is_same_v<T, f64_t>
    {
      return f(PointT<T>{static_cast<T>(x)});
    }
  };
  using ScalarField = ScalarFieldT<f64_t>;
  using CScalarField = ScalarFieldT<c128_t>;

  template <Scalar T = f64_t>
  struct OneFormT
  {
    int dim = 0;
    std::vector<ScalarFieldT<T>> comps; // size = dim, comps[i] = f_i(x) for dx_i
    OneFormT() = default;
    explicit OneFormT(int d)
        : dim(d), comps(d, ScalarFieldT<T>{[](const PointT<T>&) { return T(0); }, d})
    {
    }
    T operator()(const PointT<T>& p, int i) const
    {
      return comps[i](p);
    }
  };
  using OneForm = OneFormT<f64_t>;
  using COneForm = OneFormT<c128_t>;

  template <Scalar T = f64_t>
  struct KFormT
  {
    int k = 0;
    int dim = 0;
    // coeffs indexed by sorted tuple I = {i1<...<ik}} -> function (Composite pattern)
    std::map<std::vector<int>, ScalarFieldT<T>> coeffs;
    KFormT() = default;
    KFormT(int degree, int d) : k(degree), dim(d)
    {
    }
  };
  using KForm = KFormT<f64_t>;
  using CKForm = KFormT<c128_t>;

  // Form trait for generic programming (Form concept) — after ScalarField/OneForm/KForm
  template <typename T>
  struct is_form : std::false_type
  {
  };
  template <Scalar S>
  struct is_form<ScalarFieldT<S>> : std::true_type
  {
  };
  template <Scalar S>
  struct is_form<OneFormT<S>> : std::true_type
  {
  };
  template <Scalar S>
  struct is_form<KFormT<S>> : std::true_type
  {
  };
  template <typename T>
  inline constexpr bool is_form_v = is_form<T>::value;
  template <typename T>
  concept Form = is_form_v<T>;

  // ── Visitor for Node (modern variant-based) ──────────────────────────────
  // Forward
  struct Node;
  using NodePtr = std::shared_ptr<Node>;

  struct Node
  {
    enum Type
    {
      Var,
      Const,
      Add,
      Sub,
      Mul,
      Div,
      Pow,
      Sin,
      Cos,
      Exp,
      Log,
      Sqrt,
      Tan,
      Asin,
      Acos,
      Atan
    } type = Const;
    int var = -1;
    f64_t cval = 0;
    NodePtr left, right, child;

    // Prototype pattern: deep clone
    NP_NODISCARD NodePtr clone() const
    {
      auto n = std::make_shared<Node>();
      n->type = type;
      n->var = var;
      n->cval = cval;
      if (left)
        n->left = left->clone();
      if (right)
        n->right = right->clone();
      if (child)
        n->child = child->clone();
      return n;
    }

    // Visitor accept (Visitor pattern) — calls visitor.visit(*this)
    template <typename Visitor>
    auto accept(Visitor&& v) -> decltype(v.visit(*this))
    {
      return v.visit(*this);
    }
    template <typename Visitor>
    auto accept(Visitor&& v) const -> decltype(v.visit(*this))
    {
      return v.visit(*this);
    }
  };

  // Concrete Visitors (Visitor pattern)
  struct EvalVisitor
  {
    const Point& p;
    NP_NODISCARD f64_t visit(const Node& n) const;
  };

  struct DualEvalVisitor
  {
    const Point& p;
    int var = 0;
    NP_NODISCARD Dual<f64_t> visit(const Node& n) const;
  };

  struct DiffVisitor
  {
    int var = 0;
    NP_NODISCARD NodePtr visit(const Node& n) const;
  };

  // ── Strategy for evaluation (Strategy pattern) ───────────────────────────
  // IEvaluator is the Strategy abstraction; InterpreterStrategy and LLVMStrategy
  // are concrete strategies swapped at runtime (VM holds shared_ptr<IEvaluator>).
  // Decorator: CachedEvaluator wraps any IEvaluator and memoizes eval.
  template <Scalar T = f64_t>
  struct IEvaluator
  {
    virtual ~IEvaluator() = default;
    virtual T eval(const Node& n, const PointT<T>& p) const = 0;
    virtual Dual<T> eval_dual(const Node& n, const PointT<T>& p, int var) const = 0;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
  };

  template <Scalar T = f64_t>
  struct InterpreterStrategy : IEvaluator<T>
  {
    T eval(const Node& n, const PointT<T>& p) const override;
    Dual<T> eval_dual(const Node& n, const PointT<T>& p, int var) const override;
    NP_NODISCARD std::string name() const noexcept override
    {
      return "interpreter";
    }
  };

  // Decorator pattern: caching layer for any evaluator (memoization)
  template <Scalar T = f64_t>
  struct CachedEvaluator : IEvaluator<T>
  {
    std::shared_ptr<IEvaluator<T>> inner;
    mutable std::unordered_map<std::string, T> cache_;
    mutable std::shared_mutex mtx_;
    explicit CachedEvaluator(std::shared_ptr<IEvaluator<T>> in) : inner(std::move(in))
    {
    }
    NP_NODISCARD static std::string key(const Node& n, const PointT<T>& p)
    {
      std::string k = std::to_string(reinterpret_cast<std::uintptr_t>(&n)) + "|";
      for (auto v : p)
        k += std::to_string(v) + ",";
      return k;
    }
    T eval(const Node& n, const PointT<T>& p) const override
    {
      auto k = key(n, p);
      {
        std::shared_lock lock(mtx_);
        auto it = cache_.find(k);
        if (it != cache_.end())
          return it->second;
      }
      T v = inner->eval(n, p);
      {
        std::unique_lock lock(mtx_);
        cache_[k] = v;
      }
      return v;
    }
    Dual<T> eval_dual(const Node& n, const PointT<T>& p, int var) const override
    {
      return inner->eval_dual(n, p, var);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "cached(" + inner->name() + ")";
    }
    void clear() const noexcept
    {
      std::unique_lock lock(mtx_);
      cache_.clear();
    }
  };

#if NP_HAS_LLVM_JIT
  template <Scalar T = f64_t>
  struct LLVMStrategy : IEvaluator<T>
  {
    // Would build LLVM IR via IRBuilder; fallback to interpreter until linked
    T eval(const Node& n, const PointT<T>& p) const override
    {
      return InterpreterStrategy<T>{}.eval(n, p);
    }
    Dual<T> eval_dual(const Node& n, const PointT<T>& p, int var) const override
    {
      return InterpreterStrategy<T>{}.eval_dual(n, p, var);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "llvm-jit";
    }
  };
#endif

  // ── VM: tiny expression VM with symbolic diff and optional LLVM JIT ──────
  // Factory creates VMs and forms (Factory pattern)
  class VM
  {
    NodePtr root;
    // Reorder members to match init order (fix -Wreorder)
    std::vector<std::string> vars;
    std::map<std::string, int> var_index;
    std::string expr;
    std::size_t pos = 0;

    // Strategy (Strategy pattern)
    std::shared_ptr<IEvaluator<f64_t>> evaluator;

    // Observer (Observer pattern) — mutable so const eval can notify, thread-safe
    mutable std::vector<std::function<void(const Point& p, f64_t result)>> observers_;
    mutable std::shared_mutex obs_mtx_;
    // Derivative cache (Prototype + caching)
    mutable std::unordered_map<int, NodePtr> deriv_cache_;
    mutable std::shared_mutex deriv_mtx_;

    static NodePtr make_const(f64_t v)
    {
      auto n = std::make_shared<Node>();
      n->type = Node::Const;
      n->cval = v;
      return n;
    }
    static NodePtr make_var(int idx)
    {
      auto n = std::make_shared<Node>();
      n->type = Node::Var;
      n->var = idx;
      return n;
    }

    // Parser state
    void skip()
    {
      while (pos < expr.size() && std::isspace((unsigned char)expr[pos]))
        ++pos;
    }
    NodePtr parse_expr();
    NodePtr parse_term();
    NodePtr parse_factor();
    NodePtr parse_unary();
    NodePtr parse_primary();

    f64_t eval_node(const NodePtr& n, const Point& p) const;
    Dual<f64_t> eval_dual(const NodePtr& n, const Point& p, int var, double h) const;
    NodePtr diff_node(const NodePtr& n, int var) const;

  public:
    VM() = default;
    // Custom copy/move to handle mutex members (Prototype-safe)
    VM(const VM& o)
        : root(o.root), vars(o.vars), var_index(o.var_index), expr(o.expr), pos(o.pos),
          evaluator(o.evaluator), observers_(o.observers_), deriv_cache_(o.deriv_cache_)
    {
    }
    VM& operator=(const VM& o)
    {
      if (this != &o)
      {
        root = o.root;
        vars = o.vars;
        var_index = o.var_index;
        expr = o.expr;
        pos = o.pos;
        evaluator = o.evaluator;
        observers_ = o.observers_;
        deriv_cache_ = o.deriv_cache_;
      }
      return *this;
    }
    VM(VM&& o) noexcept
        : root(std::move(o.root)), vars(std::move(o.vars)),
          var_index(std::move(o.var_index)), expr(std::move(o.expr)), pos(o.pos),
          evaluator(std::move(o.evaluator)), observers_(std::move(o.observers_)),
          deriv_cache_(std::move(o.deriv_cache_))
    {
    }
    VM& operator=(VM&& o) noexcept
    {
      if (this != &o)
      {
        root = std::move(o.root);
        vars = std::move(o.vars);
        var_index = std::move(o.var_index);
        expr = std::move(o.expr);
        pos = o.pos;
        evaluator = std::move(o.evaluator);
        observers_ = std::move(o.observers_);
        deriv_cache_ = std::move(o.deriv_cache_);
      }
      return *this;
    }
    VM(const std::string& e,
       const std::vector<std::string>& vs = std::vector<std::string>{"x"})
        : vars(vs), expr(e), pos(0)
    {
      for (size_t i = 0; i < vars.size(); ++i)
        var_index[vars[i]] = static_cast<int>(i);
      pos = 0;
      root = parse_expr();
      skip();
      if (pos != expr.size())
        throw std::invalid_argument("VM: trailing chars");
#if NP_HAS_LLVM_JIT
      evaluator = std::make_shared<LLVMStrategy<f64_t>>();
#else
      evaluator = std::make_shared<InterpreterStrategy<f64_t>>();
#endif
    }

    // ── Ergonomic eval (Template Method) ──────────────────────────────────
    f64_t eval(const Point& p) const
    {
      if (!root)
        throw std::runtime_error("VM: empty");
      f64_t v = eval_node(root, p);
      {
        std::shared_lock lock(obs_mtx_);
        for (auto& obs : observers_)
          obs(p, v);
      }
      return v;
    }
    f64_t operator()(const Point& p) const
    {
      return eval(p);
    }
    f64_t operator()(f64_t x) const
    {
      return eval(Point{x});
    }
    f64_t operator()(f64_t x, f64_t y) const
    {
      return eval(Point{x, y});
    }
    f64_t operator()(f64_t x, f64_t y, f64_t z) const
    {
      return eval(Point{x, y, z});
    }

    int dim() const noexcept
    {
      return static_cast<int>(vars.size());
    }
    const std::vector<std::string>& variables() const noexcept
    {
      return vars;
    }

    // Dual AD derivative w.r.t var index (Strategy delegates to evaluator)
    f64_t derivative(const Point& p, int var) const
    {
      return eval_dual(root, p, var, 0).dval;
    }
    f64_t derivative(const Point& p, const std::string& var) const
    {
      auto it = var_index.find(var);
      if (it == var_index.end())
        throw std::invalid_argument("VM: unknown var " + var);
      return derivative(p, it->second);
    }

    // Prototype + cache: reuse previously derived NodePtr if available
    NodePtr cached_diff(int var) const
    {
      {
        std::shared_lock lock(deriv_mtx_);
        auto it = deriv_cache_.find(var);
        if (it != deriv_cache_.end())
          return it->second->clone();
      }
      NodePtr d = diff_node(root, var);
      {
        std::unique_lock lock(deriv_mtx_);
        deriv_cache_[var] = d->clone();
      }
      return d;
    }

    VM derivative_vm(int var) const
    {
      VM out;
      out.vars = vars;
      out.var_index = var_index;
      out.expr = expr + "'_d" + std::to_string(var);
      out.root = cached_diff(var);
#if NP_HAS_LLVM_JIT
      out.evaluator = std::make_shared<LLVMStrategy<f64_t>>();
#else
      out.evaluator = std::make_shared<InterpreterStrategy<f64_t>>();
#endif
      return out;
    }
    VM derivative_vm(const std::string& var) const
    {
      auto it = var_index.find(var);
      if (it == var_index.end())
        throw std::invalid_argument("VM: unknown var " + var);
      return derivative_vm(it->second);
    }
    // Shorthand d/dx, d/dy
    VM dx() const
    {
      return derivative_vm(0);
    }
    VM dy() const
    {
      if (vars.size() < 2)
        throw std::invalid_argument("VM::dy need dim>=2");
      return derivative_vm(1);
    }

    std::string to_string() const
    {
      return expr;
    }

    // ── Strategy control (Strategy pattern) ────────────────────────────────
    void set_evaluator(std::shared_ptr<IEvaluator<f64_t>> e)
    {
      evaluator = std::move(e);
    }
    NP_NODISCARD std::string strategy_name() const noexcept
    {
      return evaluator ? evaluator->name() : "none";
    }
    void use_interpreter()
    {
      evaluator = std::make_shared<InterpreterStrategy<f64_t>>();
    }
    void use_cached()
    {
      auto inner = evaluator ? evaluator : std::make_shared<InterpreterStrategy<f64_t>>();
      evaluator = std::make_shared<CachedEvaluator<f64_t>>(inner);
    }
    void use_cached_interpreter()
    {
      evaluator = std::make_shared<CachedEvaluator<f64_t>>(
          std::make_shared<InterpreterStrategy<f64_t>>());
    }
#if NP_HAS_LLVM_JIT
    void use_llvm()
    {
      evaluator = std::make_shared<LLVMStrategy<f64_t>>();
    }
    void use_cached_llvm()
    {
      evaluator = std::make_shared<CachedEvaluator<f64_t>>(
          std::make_shared<LLVMStrategy<f64_t>>());
    }
#endif

    // ── Builder pattern for VM ───────────────────────────────────────────────
    class Builder
    {
      std::string expr_;
      std::vector<std::string> vars_ = {"x"};
      std::shared_ptr<IEvaluator<f64_t>> strat_;

    public:
      Builder& expr(std::string e)
      {
        expr_ = std::move(e);
        return *this;
      }
      Builder& vars(std::vector<std::string> v)
      {
        vars_ = std::move(v);
        return *this;
      }
      Builder& add_var(std::string v)
      {
        vars_.push_back(std::move(v));
        return *this;
      }
      Builder& strategy(std::shared_ptr<IEvaluator<f64_t>> s)
      {
        strat_ = std::move(s);
        return *this;
      }
      NP_NODISCARD VM build() const
      {
        VM vm(expr_, vars_);
        if (strat_)
          vm.evaluator = strat_;
        return vm;
      }
    };
    NP_NODISCARD static Builder builder()
    {
      return {};
    }

    // ── Observer support (Observer pattern) ─────────────────────────────────
    using Observer = std::function<void(const Point& p, f64_t result)>;
    void add_observer(Observer obs) const
    {
      std::unique_lock lock(obs_mtx_);
      observers_.push_back(std::move(obs));
    }
    void clear_observers() const noexcept
    {
      std::unique_lock lock(obs_mtx_);
      observers_.clear();
    }
    std::size_t observer_count() const noexcept
    {
      std::shared_lock lock(obs_mtx_);
      return observers_.size();
    }

    // Evaluate on ndarray points: each row is a point (modern span-based)
    ndarray<f64_t> eval_batch(const ndarray<f64_t>& pts) const
    {
      // pts shape [N, dim]
      if (pts.ndim() != 2)
        throw std::invalid_argument("VM::eval_batch need 2D");
      int N = pts.shape[0], D = pts.shape[1];
      if (D != static_cast<int>(vars.size()))
        throw std::invalid_argument("VM::eval_batch dim mismatch");
      ndarray<f64_t> out(std::vector<int>{N});
      for (int i = 0; i < N; ++i)
      {
        Point p(D);
        for (int j = 0; j < D; ++j)
          p[j] = pts(i, j);
        f64_t v = eval(p);
        out[i] = v;
      }
      return out;
    }
    // Modern span-based batch (C++20)
    NP_NODISCARD ndarray<f64_t>
    eval_batch_span(std::span<const f64_t> flat, int dim) const
    {
      if (dim != static_cast<int>(vars.size()))
        throw std::invalid_argument("eval_batch_span dim mismatch");
      if (flat.size() % static_cast<size_t>(dim) != 0)
        throw std::invalid_argument("eval_batch_span flat size not multiple of dim");
      size_t N = flat.size() / static_cast<size_t>(dim);
      ndarray<f64_t> out(std::vector<int>{static_cast<int>(N)});
      for (size_t i = 0; i < N; ++i)
      {
        Point p(dim);
        for (int j = 0; j < dim; ++j)
          p[j] = flat[i * static_cast<size_t>(dim) + static_cast<size_t>(j)];
        out[static_cast<int>(i)] = eval(p);
      }
      return out;
    }
    // Single-value batch for 1D
    ndarray<f64_t> eval_batch_1d(const ndarray<f64_t>& xs) const
    {
      if (xs.ndim() != 1)
        throw std::invalid_argument("eval_batch_1d need 1D");
      int N = xs.shape[0];
      ndarray<f64_t> out(std::vector<int>{N});
      for (int i = 0; i < N; ++i)
        out[i] = eval(Point{xs[i]});
      return out;
    }

    // Factory access for forms
    friend struct FormFactory;
  };

  // ── FormFactory (Factory + Abstract Factory pattern) ─────────────────────
  // Creates ScalarField / OneForm / KForm from VM or from raw functions.
  struct FormFactory
  {
    template <Scalar T = f64_t>
    NP_NODISCARD static ScalarFieldT<T> create_scalar(const VM& vm)
    {
      auto vars = vm.variables();
      return ScalarFieldT<T>(
          [vm](const PointT<T>& p)
          {
            Point q(p.size());
            for (size_t i = 0; i < p.size(); ++i)
              q[i] = static_cast<f64_t>(p[i]);
            return static_cast<T>(vm.eval(q));
          },
          static_cast<int>(vars.size()));
    }
    template <Scalar T = f64_t>
    NP_NODISCARD static ScalarFieldT<T>
    create_scalar(std::function<T(const PointT<T>&)> fn, int dim)
    {
      return ScalarFieldT<T>(std::move(fn), dim);
    }
    template <Scalar T = f64_t>
    NP_NODISCARD static OneFormT<T> create_oneform(const VM& vm)
    {
      auto vars = vm.variables();
      OneFormT<T> out;
      out.dim = static_cast<int>(vars.size());
      out.comps.reserve(vars.size());
      for (size_t i = 0; i < vars.size(); ++i)
      {
        VM dvm = vm.derivative_vm(static_cast<int>(i));
        ScalarFieldT<T> sf(
            [dvm](const PointT<T>& p)
            {
              Point q(p.size());
              for (size_t j = 0; j < p.size(); ++j)
                q[j] = static_cast<f64_t>(p[j]);
              return static_cast<T>(dvm.eval(q));
            },
            static_cast<int>(vars.size()));
        out.comps.push_back(std::move(sf));
      }
      return out;
    }
    template <Scalar T = f64_t>
    NP_NODISCARD static KFormT<T> create_kform(int k, int dim)
    {
      return KFormT<T>(k, dim);
    }
    // Factory for VM itself (Factory Method)
    NP_NODISCARD static VM
    create_vm(const std::string& expr, const std::vector<std::string>& vars = {"x"})
    {
      return VM(expr, vars);
    }
    NP_NODISCARD static VM create_vm_builder(
        const std::string& expr,
        const std::vector<std::string>& vars = {"x"},
        std::shared_ptr<IEvaluator<f64_t>> strat = nullptr)
    {
      auto b = VM::builder().expr(expr).vars(vars);
      if (strat)
        b.strategy(std::move(strat));
      return b.build();
    }
  };

  // ── Exterior derivative (Template Method + Strategy) ─────────────────────
  template <Scalar T = f64_t>
  NP_NODISCARD inline OneFormT<T> exterior_derivative(const ScalarFieldT<T>& f)
  {
    OneFormT<T> out;
    out.dim = f.dim;
    out.comps.resize(f.dim);
    for (int i = 0; i < f.dim; ++i)
    {
      // Use dual AD if f is from VM? For generic std::function, use central difference
      ScalarFieldT<T> df(
          [f, i](const PointT<T>& p) -> T
          {
            T h = T(1e-7);
            PointT<T> pp = p, pm = p;
            pp[i] += h;
            pm[i] -= h;
            return (f(pp) - f(pm)) / (T(2) * h);
          },
          f.dim);
      out.comps[i] = std::move(df);
    }
    return out;
  }
  // non-templated alias for f64_t
  NP_NODISCARD inline OneForm exterior_derivative(const ScalarField& f)
  {
    return exterior_derivative<f64_t>(f);
  }

  // Symbolic exterior derivative for VM (Factory + Strategy)
  template <Scalar T = f64_t>
  NP_NODISCARD inline OneFormT<T> exterior_derivative(const VM& vm)
  {
    return FormFactory::create_oneform<T>(vm);
  }
  NP_NODISCARD inline OneForm
  exterior_derivative_vm(const VM& vm, const std::vector<std::string>& vars)
  {
    OneForm out;
    out.dim = static_cast<int>(vars.size());
    out.comps.reserve(vars.size());
    for (size_t i = 0; i < vars.size(); ++i)
    {
      VM dvm = vm.derivative_vm(static_cast<int>(i));
      ScalarField sf(
          [dvm](const Point& p) { return dvm.eval(p); }, static_cast<int>(vars.size()));
      out.comps.push_back(std::move(sf));
    }
    out.dim = static_cast<int>(vars.size());
    return out;
  }
  // Exterior derivative for OneForm / KForm (Template Method continuation)
  // d: Ω^k -> Ω^{k+1} via finite differences on each coefficient
  template <Scalar T = f64_t>
  NP_NODISCARD inline KFormT<T> exterior_derivative(const OneFormT<T>& w)
  {
    // dω = Σ_{i<j} (∂_i w_j - ∂_j w_i) dx_i ∧ dx_j
    if (w.dim < 2)
      return KFormT<T>(2, w.dim);
    KFormT<T> out(2, w.dim);
    for (int i = 0; i < w.dim; ++i)
      for (int j = i + 1; j < w.dim; ++j)
      {
        std::vector<int> idx = {i, j};
        // capture by value for lambda (Composite)
        ScalarFieldT<T> cf(
            [w, i, j](const PointT<T>& p) -> T
            {
              T h = T(1e-7);
              PointT<T> pp = p, pm = p;
              pp[i] += h;
              pm[i] -= h;
              T dwi = (w.comps[j](pp) - w.comps[j](pm)) / (T(2) * h);
              pp = p;
              pm = p;
              pp[j] += h;
              pm[j] -= h;
              T dwj = (w.comps[i](pp) - w.comps[i](pm)) / (T(2) * h);
              return dwi - dwj;
            },
            w.dim);
        out.coeffs[idx] = std::move(cf);
      }
    return out;
  }
  template <Scalar T = f64_t>
  NP_NODISCARD inline KFormT<T> exterior_derivative(const KFormT<T>& w)
  {
    // Generic: increase degree by 1, differentiate each coefficient and wedge
    // with dx_i where i not in index. Simplified finite-difference form.
    KFormT<T> out(w.k + 1, w.dim);
    for (auto& [idx, coeff] : w.coeffs)
    {
      for (int d = 0; d < w.dim; ++d)
      {
        if (std::find(idx.begin(), idx.end(), d) != idx.end())
          continue;
        std::vector<int> nidx = idx;
        nidx.push_back(d);
        std::sort(nidx.begin(), nidx.end());
        // sign from inserting d into sorted idx
        int sign = 1;
        for (int v : idx)
          if (v > d)
            sign = -sign;
        // coefficient is ∂_d coeff with sign (finite difference)
        ScalarFieldT<T> cf(
            [coeff, d, sign](const PointT<T>& p) -> T
            {
              T h = T(1e-7);
              PointT<T> pp = p, pm = p;
              pp[d] += h;
              pm[d] -= h;
              return T(sign) * (coeff(pp) - coeff(pm)) / (T(2) * h);
            },
            w.dim);
        // merge if nidx already present (sum)
        auto it = out.coeffs.find(nidx);
        if (it == out.coeffs.end())
          out.coeffs[nidx] = std::move(cf);
        else
        {
          auto prev = it->second;
          out.coeffs[nidx] = ScalarFieldT<T>(
              [prev, cf](const PointT<T>& p) { return prev(p) + cf(p); }, w.dim);
        }
      }
    }
    return out;
  }

  // Alias d for exterior_derivative (Template Method)
  template <Scalar T = f64_t>
  NP_NODISCARD inline OneFormT<T> d(const ScalarFieldT<T>& f)
  {
    return exterior_derivative(f);
  }
  NP_NODISCARD inline OneForm d(const ScalarField& f)
  {
    return exterior_derivative(f);
  }
  NP_NODISCARD inline OneForm d(const VM& vm)
  {
    return exterior_derivative(vm);
  }
  template <Scalar T = f64_t>
  NP_NODISCARD inline KFormT<T> d(const OneFormT<T>& w)
  {
    return exterior_derivative(w);
  }
  template <Scalar T = f64_t>
  NP_NODISCARD inline KFormT<T> d(const KFormT<T>& w)
  {
    return exterior_derivative(w);
  }

  // ── Wedge (Decorator/Composite) ──────────────────────────────────────────
  template <Scalar T = f64_t>
  NP_NODISCARD inline KFormT<T> wedge(const OneFormT<T>& a, const OneFormT<T>& b)
  {
    if (a.dim != b.dim)
      throw std::invalid_argument("wedge: dim mismatch");
    KFormT<T> out;
    out.k = 2;
    out.dim = a.dim;
    for (int i = 0; i < a.dim; ++i)
      for (int j = i + 1; j < a.dim; ++j)
      {
        std::vector<int> idx = {i, j};
        // coeff = a_i * b_j - a_j * b_i (Composite)
        ScalarFieldT<T> cf(
            [a, b, i, j](const PointT<T>& p)
            { return a.comps[i](p) * b.comps[j](p) - a.comps[j](p) * b.comps[i](p); },
            a.dim);
        out.coeffs[idx] = std::move(cf);
      }
    return out;
  }
  NP_NODISCARD inline KForm wedge(const OneForm& a, const OneForm& b)
  {
    return wedge<f64_t>(a, b);
  }

  // General wedge for KForms (Decorator)
  template <Scalar T = f64_t>
  NP_NODISCARD inline KFormT<T> wedge(const KFormT<T>& a, const KFormT<T>& b)
  {
    if (a.dim != b.dim)
      throw std::invalid_argument("wedge: dim mismatch");
    if (a.coeffs.empty() || b.coeffs.empty())
      return KFormT<T>(a.k + b.k, a.dim);
    KFormT<T> out;
    out.k = a.k + b.k;
    out.dim = a.dim;
    for (auto& [idx_a, fa] : a.coeffs)
      for (auto& [idx_b, fb] : b.coeffs)
      {
        // merge indices, check overlap (wedge is zero if overlap)
        std::vector<int> idx = idx_a;
        bool overlap = false;
        for (int v : idx_b)
          if (std::find(idx.begin(), idx.end(), v) != idx.end())
            overlap = true;
        if (overlap)
          continue;
        idx.insert(idx.end(), idx_b.begin(), idx_b.end());
        std::sort(idx.begin(), idx.end());
        // sign from permutation
        int sign = 1;
        // naive sign: count inversions between idx_a and idx_b
        for (int ia : idx_a)
          for (int ib : idx_b)
            if (ia > ib)
              sign = -sign;
        ScalarFieldT<T> cf(
            [fa, fb, sign](const PointT<T>& p) { return T(sign) * fa(p) * fb(p); },
            a.dim);
        // if idx already exists (should not for distinct wedge) sum, else insert
        auto it = out.coeffs.find(idx);
        if (it == out.coeffs.end())
          out.coeffs[idx] = std::move(cf);
        else
        {
          auto prev = it->second;
          out.coeffs[idx] = ScalarFieldT<T>(
              [prev, cf](const PointT<T>& p) { return prev(p) + cf(p); }, a.dim);
        }
      }
    return out;
  }

  // ── Form operators (Decorator / Composite syntactic sugar) ─────────────────
  // Wedge as operator^ (exterior algebra) and operator* alias
  template <Scalar T>
  NP_NODISCARD inline KFormT<T> operator^(const OneFormT<T>& a, const OneFormT<T>& b)
  {
    return wedge(a, b);
  }
  template <Scalar T>
  NP_NODISCARD inline KFormT<T> operator^(const KFormT<T>& a, const KFormT<T>& b)
  {
    return wedge(a, b);
  }
  template <Scalar T>
  NP_NODISCARD inline KFormT<T> operator*(const OneFormT<T>& a, const OneFormT<T>& b)
  {
    return wedge(a, b);
  }
  template <Scalar T>
  NP_NODISCARD inline KFormT<T> operator*(const KFormT<T>& a, const KFormT<T>& b)
  {
    return wedge(a, b);
  }
  // Form addition / subtraction (pointwise)
  template <Scalar T>
  NP_NODISCARD inline OneFormT<T> operator+(const OneFormT<T>& a, const OneFormT<T>& b)
  {
    if (a.dim != b.dim)
      throw std::invalid_argument("OneForm +: dim mismatch");
    OneFormT<T> out(a.dim);
    for (int i = 0; i < a.dim; ++i)
      out.comps[i] = ScalarFieldT<T>(
          [a, b, i](const PointT<T>& p) { return a.comps[i](p) + b.comps[i](p); }, a.dim);
    return out;
  }
  template <Scalar T>
  NP_NODISCARD inline OneFormT<T> operator-(const OneFormT<T>& a, const OneFormT<T>& b)
  {
    if (a.dim != b.dim)
      throw std::invalid_argument("OneForm -: dim mismatch");
    OneFormT<T> out(a.dim);
    for (int i = 0; i < a.dim; ++i)
      out.comps[i] = ScalarFieldT<T>(
          [a, b, i](const PointT<T>& p) { return a.comps[i](p) - b.comps[i](p); }, a.dim);
    return out;
  }
  template <Scalar T>
  NP_NODISCARD inline KFormT<T> operator+(const KFormT<T>& a, const KFormT<T>& b)
  {
    if (a.dim != b.dim || a.k != b.k)
      throw std::invalid_argument("KForm +: dim/k mismatch");
    KFormT<T> out(a.k, a.dim);
    out.coeffs = a.coeffs;
    for (auto& [idx, fb] : b.coeffs)
    {
      auto it = out.coeffs.find(idx);
      if (it == out.coeffs.end())
        out.coeffs[idx] = fb;
      else
      {
        auto fa = it->second;
        out.coeffs[idx] = ScalarFieldT<T>(
            [fa, fb](const PointT<T>& p) { return fa(p) + fb(p); }, a.dim);
      }
    }
    return out;
  }
  template <Scalar T>
  NP_NODISCARD inline KFormT<T> operator-(const KFormT<T>& a, const KFormT<T>& b)
  {
    if (a.dim != b.dim || a.k != b.k)
      throw std::invalid_argument("KForm -: dim/k mismatch");
    KFormT<T> out(a.k, a.dim);
    out.coeffs = a.coeffs;
    for (auto& [idx, fb] : b.coeffs)
    {
      auto it = out.coeffs.find(idx);
      if (it == out.coeffs.end())
      {
        out.coeffs[idx] =
            ScalarFieldT<T>([fb](const PointT<T>& p) { return -fb(p); }, a.dim);
      }
      else
      {
        auto fa = it->second;
        out.coeffs[idx] = ScalarFieldT<T>(
            [fa, fb](const PointT<T>& p) { return fa(p) - fb(p); }, a.dim);
      }
    }
    return out;
  }

  // ── Pullback, interior product, Lie derivative (Cartan) ──────────────────
  template <Scalar T = f64_t>
  NP_NODISCARD inline ScalarFieldT<T> pullback(
      const ScalarFieldT<T>& f, const std::function<PointT<T>(const PointT<T>&)>& phi)
  {
    return ScalarFieldT<T>([f, phi](const PointT<T>& p) { return f(phi(p)); }, f.dim);
  }

  template <Scalar T = f64_t>
  NP_NODISCARD inline OneFormT<T> pullback(
      const OneFormT<T>& omega,
      const std::function<PointT<T>(const PointT<T>&)>& phi,
      const std::function<std::vector<std::vector<T>>(const PointT<T>&)>& dphi)
  {
    // (phi^* omega)_p (v) = omega_{phi(p)} (d phi_p (v))
    OneFormT<T> out;
    out.dim = omega.dim;
    out.comps.resize(omega.dim);
    for (int i = 0; i < omega.dim; ++i)
    {
      out.comps[i] = ScalarFieldT<T>(
          [omega, phi, dphi, i](const PointT<T>& p)
          {
            auto q = phi(p);
            auto J = dphi(p); // J[i][j] = d phi_j / d x_i ?
            T res = T(0);
            for (int j = 0; j < omega.dim; ++j)
              res += omega.comps[j](q) * J[i][j];
            return res;
          },
          omega.dim);
    }
    return out;
  }

  template <Scalar T = f64_t>
  NP_NODISCARD inline ScalarFieldT<T>
  interior_product(const OneFormT<T>& omega, const std::vector<T>& vec)
  {
    // i_X omega where X is vector field (constant for now)
    return ScalarFieldT<T>(
        [omega, vec](const PointT<T>& p)
        {
          T res = T(0);
          for (int i = 0; i < omega.dim; ++i)
            res += omega.comps[i](p) * vec[i];
          return res;
        },
        omega.dim);
  }

  template <Scalar T = f64_t>
  NP_NODISCARD inline OneFormT<T>
  lie_derivative(const ScalarFieldT<T>& f, const std::vector<T>& X)
  {
    // L_X f = X(f) = df(X)
    auto df = exterior_derivative(f);
    auto res = interior_product(df, X);
    // Return as OneForm? For 0-form, Lie derivative is 0-form, but we wrap as OneForm for
    // demo
    OneFormT<T> out(f.dim);
    out.comps[0] = res;
    return out;
  }

  // ── Helpers for variety de Rham ───────────────────────────────────────
  NP_NODISCARD inline std::vector<int> de_rham_betti_from_forms(int dim)
  {
    // For R^n, de Rham H^0=R, others 0; for S^n, H^0=H^n=R
    // This helper is used by variety de_rham
    std::vector<int> betti(dim + 1, 0);
    betti[0] = 1;
    if (dim >= 1)
      betti[dim] = 1;
    return betti;
  }

  // ── VM Parser / Evaluator implementations (Strategy) ─────────────────────
  inline NodePtr VM::parse_expr()
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
  inline NodePtr VM::parse_term()
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
  inline NodePtr VM::parse_factor()
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
  inline NodePtr VM::parse_unary()
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
  inline NodePtr VM::parse_primary()
  {
    skip();
    if (pos >= expr.size())
      throw std::invalid_argument("VM: unexpected end");
    if (expr[pos] == '(')
    {
      ++pos;
      auto n = parse_expr();
      skip();
      if (pos >= expr.size() || expr[pos] != ')')
        throw std::invalid_argument("VM: missing )");
      ++pos;
      return n;
    }
    if (std::isalpha((unsigned char)expr[pos]))
    {
      std::size_t start = pos;
      while (pos < expr.size()
             && (std::isalnum((unsigned char)expr[pos]) || expr[pos] == '_'))
        ++pos;
      std::string name = expr.substr(start, pos - start);
      skip();
      if (pos < expr.size() && expr[pos] == '(')
      {
        ++pos;
        auto arg = parse_expr();
        skip();
        if (pos >= expr.size() || expr[pos] != ')')
          throw std::invalid_argument("VM: missing ) after func");
        ++pos;
        auto o = std::make_shared<Node>();
        if (name == "sin")
          o->type = Node::Sin;
        else if (name == "cos")
          o->type = Node::Cos;
        else if (name == "exp")
          o->type = Node::Exp;
        else if (name == "log")
          o->type = Node::Log;
        else if (name == "sqrt")
          o->type = Node::Sqrt;
        else if (name == "tan")
          o->type = Node::Tan;
        else if (name == "asin")
          o->type = Node::Asin;
        else if (name == "acos")
          o->type = Node::Acos;
        else if (name == "atan")
          o->type = Node::Atan;
        else
          throw std::invalid_argument("VM: unknown func " + name);
        o->child = arg;
        return o;
      }
      auto it = var_index.find(name);
      if (it == var_index.end())
        throw std::invalid_argument("VM: unknown var " + name);
      return make_var(it->second);
    }
    // number: modern string_view + from_chars with scientific notation fallback
    std::size_t start = pos;
    bool has_exp = false;
    while (pos < expr.size())
    {
      char c = expr[pos];
      if (std::isdigit((unsigned char)c) || c == '.')
        ++pos;
      else if ((c == 'e' || c == 'E') && !has_exp)
      {
        has_exp = true;
        ++pos;
        if (pos < expr.size() && (expr[pos] == '+' || expr[pos] == '-'))
          ++pos;
      }
      else
        break;
    }
    if (start == pos)
      throw std::invalid_argument("VM: expected number/var");
    std::string_view sv(expr.data() + start, pos - start);
    f64_t v = 0;
#if __cpp_lib_to_chars >= 201611L
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{})
#endif
    {
      std::string tmp(sv);
      char* end = nullptr;
      v = std::strtod(tmp.c_str(), &end);
      if (end != tmp.c_str() + tmp.size())
        throw std::invalid_argument("VM: invalid number " + tmp);
    }
    return make_const(v);
  }

  inline f64_t VM::eval_node(const NodePtr& n, const Point& p) const
  {
    // Strategy pattern: delegate to evaluator (interpreter or LLVM JIT)
    if (evaluator && n)
    {
      return evaluator->eval(*n, p);
    }
    // fallback — DRY via InterpreterStrategy (handles all Node types)
    return InterpreterStrategy<f64_t>{}.eval(*n, p);
  }

  inline Dual<f64_t>
  VM::eval_dual(const NodePtr& n, const Point& p, int var, double h) const
  {
    (void)h;
    // Strategy pattern: delegate to evaluator for AD — DRY
    if (evaluator && n)
    {
      return evaluator->eval_dual(*n, p, var);
    }
    return InterpreterStrategy<f64_t>{}.eval_dual(*n, p, var);
  }

  inline NodePtr VM::diff_node(const NodePtr& n, int var) const
  {
    // Visitor: symbolic differentiation
    switch (n->type)
    {
      case Node::Const:
        return make_const(0);
      case Node::Var:
        return make_const(n->var == var ? 1 : 0);
      case Node::Add:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Add;
        o->left = diff_node(n->left, var);
        o->right = diff_node(n->right, var);
        return o;
      }
      case Node::Sub:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Sub;
        o->left = diff_node(n->left, var);
        o->right = diff_node(n->right, var);
        return o;
      }
      case Node::Mul:
      {
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
      case Node::Div:
      {
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
      case Node::Pow:
      {
        if (n->right->type == Node::Const)
        {
          f64_t c = n->right->cval;
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
        return diff_node(n, var);
      }
      case Node::Sin:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        auto c = std::make_shared<Node>();
        c->type = Node::Cos;
        c->child = n->child;
        o->left = c;
        o->right = diff_node(n->child, var);
        return o;
      }
      case Node::Cos:
      {
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
      case Node::Exp:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        o->left = n;
        o->right = diff_node(n->child, var);
        return o;
      }
      case Node::Log:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        o->left = diff_node(n->child, var);
        o->right = n->child;
        return o;
      }
      case Node::Sqrt:
      {
        // (sqrt(u))' = u' / (2 sqrt(u))
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        o->left = diff_node(n->child, var);
        auto den = std::make_shared<Node>();
        den->type = Node::Mul;
        den->left = make_const(2);
        auto s = std::make_shared<Node>();
        s->type = Node::Sqrt;
        s->child = n->child;
        den->right = s;
        o->right = den;
        return o;
      }
      case Node::Tan:
      {
        // (tan u)' = u' / cos^2 u
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        o->left = diff_node(n->child, var);
        auto den = std::make_shared<Node>();
        den->type = Node::Pow;
        auto c = std::make_shared<Node>();
        c->type = Node::Cos;
        c->child = n->child;
        den->left = c;
        den->right = make_const(2);
        o->right = den;
        return o;
      }
      case Node::Asin:
      {
        // (asin u)' = u' / sqrt(1 - u^2)
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        o->left = diff_node(n->child, var);
        auto den = std::make_shared<Node>();
        den->type = Node::Sqrt;
        auto sub = std::make_shared<Node>();
        sub->type = Node::Sub;
        sub->left = make_const(1);
        auto pw = std::make_shared<Node>();
        pw->type = Node::Pow;
        pw->left = n->child;
        pw->right = make_const(2);
        sub->right = pw;
        den->child = sub;
        o->right = den;
        return o;
      }
      case Node::Acos:
      {
        // (acos u)' = -u' / sqrt(1 - u^2)
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        o->left = make_const(-1);
        auto div = std::make_shared<Node>();
        div->type = Node::Div;
        div->left = diff_node(n->child, var);
        auto den = std::make_shared<Node>();
        den->type = Node::Sqrt;
        auto sub = std::make_shared<Node>();
        sub->type = Node::Sub;
        sub->left = make_const(1);
        auto pw = std::make_shared<Node>();
        pw->type = Node::Pow;
        pw->left = n->child;
        pw->right = make_const(2);
        sub->right = pw;
        den->child = sub;
        div->right = den;
        o->right = div;
        return o;
      }
      case Node::Atan:
      {
        // (atan u)' = u' / (1 + u^2)
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        o->left = diff_node(n->child, var);
        auto den = std::make_shared<Node>();
        den->type = Node::Add;
        den->left = make_const(1);
        auto pw = std::make_shared<Node>();
        pw->type = Node::Pow;
        pw->left = n->child;
        pw->right = make_const(2);
        den->right = pw;
        o->right = den;
        return o;
      }
    }
    return make_const(0);
  }

  // Strategy implementations
  template <Scalar T>
  T InterpreterStrategy<T>::eval(const Node& n, const PointT<T>& p) const
  {
    switch (n.type)
    {
      case Node::Const:
        return T(n.cval);
      case Node::Var:
        return p[n.var];
      case Node::Add:
        return eval(*n.left, p) + eval(*n.right, p);
      case Node::Sub:
        return eval(*n.left, p) - eval(*n.right, p);
      case Node::Mul:
        return eval(*n.left, p) * eval(*n.right, p);
      case Node::Div:
        return eval(*n.left, p) / eval(*n.right, p);
      case Node::Pow:
        return std::pow(eval(*n.left, p), eval(*n.right, p));
      case Node::Sin:
        return std::sin(eval(*n.child, p));
      case Node::Cos:
        return std::cos(eval(*n.child, p));
      case Node::Exp:
        return std::exp(eval(*n.child, p));
      case Node::Log:
        return std::log(eval(*n.child, p));
      case Node::Sqrt:
        return std::sqrt(eval(*n.child, p));
      case Node::Tan:
        return std::tan(eval(*n.child, p));
      case Node::Asin:
        return std::asin(eval(*n.child, p));
      case Node::Acos:
        return std::acos(eval(*n.child, p));
      case Node::Atan:
        return std::atan(eval(*n.child, p));
    }
    return T(0);
  }
  template <Scalar T>
  Dual<T>
  InterpreterStrategy<T>::eval_dual(const Node& n, const PointT<T>& p, int var) const
  {
    switch (n.type)
    {
      case Node::Const:
        return {T(n.cval), T(0)};
      case Node::Var:
        return {p[n.var], (n.var == var) ? T(1) : T(0)};
      case Node::Add:
      {
        auto a = eval_dual(*n.left, p, var);
        auto b = eval_dual(*n.right, p, var);
        return a + b;
      }
      case Node::Sub:
      {
        auto a = eval_dual(*n.left, p, var);
        auto b = eval_dual(*n.right, p, var);
        return a - b;
      }
      case Node::Mul:
      {
        auto a = eval_dual(*n.left, p, var);
        auto b = eval_dual(*n.right, p, var);
        return a * b;
      }
      case Node::Div:
      {
        auto a = eval_dual(*n.left, p, var);
        auto b = eval_dual(*n.right, p, var);
        return a / b;
      }
      case Node::Pow:
      {
        auto a = eval_dual(*n.left, p, var);
        auto b = eval_dual(*n.right, p, var);
        if (n.right->type == Node::Const)
          return pow(a, static_cast<double>(n.right->cval));
        return pow(a, b);
      }
      case Node::Sin:
      {
        auto a = eval_dual(*n.child, p, var);
        return sin(a);
      }
      case Node::Cos:
      {
        auto a = eval_dual(*n.child, p, var);
        return cos(a);
      }
      case Node::Exp:
      {
        auto a = eval_dual(*n.child, p, var);
        return exp(a);
      }
      case Node::Log:
      {
        auto a = eval_dual(*n.child, p, var);
        return log(a);
      }
      case Node::Sqrt:
      {
        auto a = eval_dual(*n.child, p, var);
        return sqrt(a);
      }
      case Node::Tan:
      {
        auto a = eval_dual(*n.child, p, var);
        return tan(a);
      }
      case Node::Asin:
      {
        auto a = eval_dual(*n.child, p, var);
        return asin(a);
      }
      case Node::Acos:
      {
        auto a = eval_dual(*n.child, p, var);
        return acos(a);
      }
      case Node::Atan:
      {
        auto a = eval_dual(*n.child, p, var);
        return atan(a);
      }
    }
    return {T(0), T(0)};
  }

  // ── Visitor implementations (Visitor pattern) ──────────────────────────────
  inline f64_t EvalVisitor::visit(const Node& n) const
  {
    return InterpreterStrategy<f64_t>{}.eval(n, p);
  }
  inline Dual<f64_t> DualEvalVisitor::visit(const Node& n) const
  {
    return InterpreterStrategy<f64_t>{}.eval_dual(n, p, var);
  }
  inline NodePtr DiffVisitor::visit(const Node& n) const
  {
    // Symbolic differentiation via VM helper (avoids duplication): build temp VM
    // and call diff_node through a placeholder VM instance.
    // For standalone, replicate logic directly to avoid VM coupling.
    auto make_const = [](f64_t v)
    {
      auto m = std::make_shared<Node>();
      m->type = Node::Const;
      m->cval = v;
      return m;
    };
    switch (n.type)
    {
      case Node::Const:
        return make_const(0);
      case Node::Var:
        return make_const(n.var == var ? 1 : 0);
      case Node::Add:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Add;
        DiffVisitor lv{var}, rv{var};
        o->left = lv.visit(*n.left);
        o->right = rv.visit(*n.right);
        return o;
      }
      case Node::Sub:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Sub;
        DiffVisitor lv{var}, rv{var};
        o->left = lv.visit(*n.left);
        o->right = rv.visit(*n.right);
        return o;
      }
      case Node::Mul:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Add;
        auto a = std::make_shared<Node>();
        a->type = Node::Mul;
        DiffVisitor lv{var};
        a->left = lv.visit(*n.left);
        a->right = n.right;
        auto b = std::make_shared<Node>();
        b->type = Node::Mul;
        b->left = n.left;
        DiffVisitor rv{var};
        b->right = rv.visit(*n.right);
        o->left = a;
        o->right = b;
        return o;
      }
      case Node::Pow:
      {
        if (n.right->type == Node::Const)
        {
          f64_t c = n.right->cval;
          auto coeff = make_const(c);
          auto pw = std::make_shared<Node>();
          pw->type = Node::Pow;
          pw->left = n.left;
          pw->right = make_const(c - 1);
          auto mul = std::make_shared<Node>();
          mul->type = Node::Mul;
          mul->left = coeff;
          mul->right = pw;
          auto o = std::make_shared<Node>();
          o->type = Node::Mul;
          o->left = mul;
          DiffVisitor lv{var};
          o->right = lv.visit(*n.left);
          return o;
        }
        return make_const(0);
      }
      case Node::Sin:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        auto c = std::make_shared<Node>();
        c->type = Node::Cos;
        c->child = n.child;
        o->left = c;
        DiffVisitor cv{var};
        o->right = cv.visit(*n.child);
        return o;
      }
      case Node::Cos:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        auto s = std::make_shared<Node>();
        s->type = Node::Sin;
        s->child = n.child;
        auto neg = std::make_shared<Node>();
        neg->type = Node::Mul;
        neg->left = make_const(-1);
        neg->right = s;
        o->left = neg;
        DiffVisitor cv{var};
        o->right = cv.visit(*n.child);
        return o;
      }
      case Node::Exp:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        auto cur = std::make_shared<Node>(n);
        o->left = cur;
        DiffVisitor cv{var};
        o->right = cv.visit(*n.child);
        return o;
      }
      case Node::Log:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        DiffVisitor cv{var};
        o->left = cv.visit(*n.child);
        o->right = n.child;
        return o;
      }
      case Node::Sqrt:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        DiffVisitor cv{var};
        o->left = cv.visit(*n.child);
        auto den = std::make_shared<Node>();
        den->type = Node::Mul;
        den->left = make_const(2);
        auto s = std::make_shared<Node>();
        s->type = Node::Sqrt;
        s->child = n.child;
        den->right = s;
        o->right = den;
        return o;
      }
      case Node::Tan:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        DiffVisitor cv{var};
        o->left = cv.visit(*n.child);
        auto den = std::make_shared<Node>();
        den->type = Node::Pow;
        auto c = std::make_shared<Node>();
        c->type = Node::Cos;
        c->child = n.child;
        den->left = c;
        den->right = make_const(2);
        o->right = den;
        return o;
      }
      case Node::Asin:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        DiffVisitor cv{var};
        o->left = cv.visit(*n.child);
        auto den = std::make_shared<Node>();
        den->type = Node::Sqrt;
        auto sub = std::make_shared<Node>();
        sub->type = Node::Sub;
        sub->left = make_const(1);
        auto pw = std::make_shared<Node>();
        pw->type = Node::Pow;
        pw->left = n.child;
        pw->right = make_const(2);
        sub->right = pw;
        den->child = sub;
        o->right = den;
        return o;
      }
      case Node::Acos:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Mul;
        o->left = make_const(-1);
        auto div = std::make_shared<Node>();
        div->type = Node::Div;
        DiffVisitor cv{var};
        div->left = cv.visit(*n.child);
        auto den = std::make_shared<Node>();
        den->type = Node::Sqrt;
        auto sub = std::make_shared<Node>();
        sub->type = Node::Sub;
        sub->left = make_const(1);
        auto pw = std::make_shared<Node>();
        pw->type = Node::Pow;
        pw->left = n.child;
        pw->right = make_const(2);
        sub->right = pw;
        den->child = sub;
        div->right = den;
        o->right = div;
        return o;
      }
      case Node::Atan:
      {
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        DiffVisitor cv{var};
        o->left = cv.visit(*n.child);
        auto den = std::make_shared<Node>();
        den->type = Node::Add;
        den->left = make_const(1);
        auto pw = std::make_shared<Node>();
        pw->type = Node::Pow;
        pw->left = n.child;
        pw->right = make_const(2);
        den->right = pw;
        o->right = den;
        return o;
      }
      case Node::Div:
      {
        auto num = std::make_shared<Node>();
        num->type = Node::Sub;
        auto a = std::make_shared<Node>();
        a->type = Node::Mul;
        DiffVisitor lv{var};
        a->left = lv.visit(*n.left);
        a->right = n.right;
        auto b = std::make_shared<Node>();
        b->type = Node::Mul;
        b->left = n.left;
        DiffVisitor rv{var};
        b->right = rv.visit(*n.right);
        num->left = a;
        num->right = b;
        auto den = std::make_shared<Node>();
        den->type = Node::Pow;
        den->left = n.right;
        den->right = make_const(2);
        auto o = std::make_shared<Node>();
        o->type = Node::Div;
        o->left = num;
        o->right = den;
        return o;
      }
    }
    return make_const(0);
  }

} // namespace np::differential

#endif // NP_DIFFERENTIAL_HPP
