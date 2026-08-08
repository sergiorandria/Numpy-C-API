/**
 * @file dtype.hpp
 * @brief NumPy-compatible data type system.
 *
 * Defines the `np::dtype` enumeration and compile-time bridges between
 * `np::dtype` values and native C++ types.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_DTYPE_HPP
#define NP_DTYPE_HPP

#include <complex>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

#include "api_macros.hpp"

namespace np {
/**
 * @brief Enumeration of NumPy-compatible data types.
 *
 * The values mirror numpy.dtype names. `string_` and `unicode_`
 * are present for API parity but have no C++ storage representation
 * (size() returns 0). `datetime64` and `timedelta64` store as
 * int64_t units with a separate `np::datetime64` unit code.
 */
enum class dtype {
  // Integer types
  int8,
  int16,
  int32,
  int64,
  uint8,
  uint16,
  uint32,
  uint64,

  // Floating-point types
  float16,
  float32,
  float64,
  longdouble,

  // Complex types
  complex64,
  complex128,
  clongdouble,

  // Boolean
  bool_,

  // String / unicode
  string_,
  unicode_,

  // Datetime
  datetime64,
  timedelta64,

  // Special
  void_,
  object_
};

// Convenience constants (same spelling as NumPy scalars).
inline constexpr dtype int8 = dtype::int8;
inline constexpr dtype int16 = dtype::int16;
inline constexpr dtype int32 = dtype::int32;
inline constexpr dtype int64 = dtype::int64;
inline constexpr dtype uint8 = dtype::uint8;
inline constexpr dtype uint16 = dtype::uint16;
inline constexpr dtype uint32 = dtype::uint32;
inline constexpr dtype uint64 = dtype::uint64;
inline constexpr dtype float16 = dtype::float16;
inline constexpr dtype float32 = dtype::float32;
inline constexpr dtype float64 = dtype::float64;
inline constexpr dtype longdouble = dtype::longdouble;
inline constexpr dtype complex64 = dtype::complex64;
inline constexpr dtype complex128 = dtype::complex128;
inline constexpr dtype clongdouble = dtype::clongdouble;
inline constexpr dtype bool_ = dtype::bool_;
inline constexpr dtype string_ = dtype::string_;
inline constexpr dtype unicode_ = dtype::unicode_;
inline constexpr dtype datetime64 = dtype::datetime64;
inline constexpr dtype timedelta64 = dtype::timedelta64;
inline constexpr dtype void_ = dtype::void_;
inline constexpr dtype object_ = dtype::object_;

namespace detail {

/**
 * @brief Maps a np::dtype value to its native C++ type.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 */
template <dtype _DtypeElement> struct _Np_type_to_cxx;

template <> struct _Np_type_to_cxx<dtype::int8> {
  using type = std::int8_t;
};
template <> struct _Np_type_to_cxx<dtype::int16> {
  using type = std::int16_t;
};
template <> struct _Np_type_to_cxx<dtype::int32> {
  using type = std::int32_t;
};
template <> struct _Np_type_to_cxx<dtype::int64> {
  using type = std::int64_t;
};
template <> struct _Np_type_to_cxx<dtype::uint8> {
  using type = std::uint8_t;
};
template <> struct _Np_type_to_cxx<dtype::uint16> {
  using type = std::uint16_t;
};
template <> struct _Np_type_to_cxx<dtype::uint32> {
  using type = std::uint32_t;
};
template <> struct _Np_type_to_cxx<dtype::uint64> {
  using type = std::uint64_t;
};
template <> struct _Np_type_to_cxx<dtype::float16> {
  using type = std::uint16_t;
};
template <> struct _Np_type_to_cxx<dtype::float32> {
  using type = float;
};
template <> struct _Np_type_to_cxx<dtype::float64> {
  using type = double;
};
template <> struct _Np_type_to_cxx<dtype::longdouble> {
  using type = long double;
};
template <> struct _Np_type_to_cxx<dtype::complex64> {
  using type = std::complex<float>;
};
template <> struct _Np_type_to_cxx<dtype::complex128> {
  using type = std::complex<double>;
};
template <> struct _Np_type_to_cxx<dtype::clongdouble> {
  using type = std::complex<long double>;
};
template <> struct _Np_type_to_cxx<dtype::bool_> {
  using type = bool;
};
template <> struct _Np_type_to_cxx<dtype::datetime64> {
  using type = std::int64_t;
};
template <> struct _Np_type_to_cxx<dtype::timedelta64> {
  using type = std::int64_t;
};

/**
 * @brief Maps a native C++ type to its np::dtype value.
 *
 * Uses a type-keyed constant expression so that aliased types
 * (e.g. std::int32_t == int on many ABIs) cannot collide.
 *
 * @tparam T  A native C++ type.
 */
template <typename T> struct cxx_to_np_type_impl {
  static constexpr dtype value =
      std::is_same_v<T, std::int8_t>                 ? dtype::int8
      : std::is_same_v<T, std::int16_t>              ? dtype::int16
      : std::is_same_v<T, std::int32_t>              ? dtype::int32
      : std::is_same_v<T, std::int64_t>              ? dtype::int64
      : std::is_same_v<T, std::uint8_t>              ? dtype::uint8
      : std::is_same_v<T, std::uint16_t>             ? dtype::uint16
      : std::is_same_v<T, std::uint32_t>             ? dtype::uint32
      : std::is_same_v<T, std::uint64_t>             ? dtype::uint64
      : std::is_same_v<T, float>                     ? dtype::float32
      : std::is_same_v<T, double>                    ? dtype::float64
      : std::is_same_v<T, long double>               ? dtype::longdouble
      : std::is_same_v<T, std::complex<float>>       ? dtype::complex64
      : std::is_same_v<T, std::complex<double>>      ? dtype::complex128
      : std::is_same_v<T, std::complex<long double>> ? dtype::clongdouble
      : std::is_same_v<T, bool>                      ? dtype::bool_
      : std::is_same_v<T, char>                      ? dtype::int8
      : std::is_same_v<T, signed char>               ? dtype::int8
      : std::is_same_v<T, unsigned char>             ? dtype::uint8
      : std::is_same_v<T, wchar_t>                   ? dtype::int32
      : std::is_same_v<T, std::size_t>
          ? (sizeof(std::size_t) == 8 ? dtype::uint64 : dtype::uint32)
      : std::is_same_v<T, std::ptrdiff_t>
          ? (sizeof(std::ptrdiff_t) == 8 ? dtype::int64 : dtype::int32)
          : dtype::void_;
};

template <typename T>
struct cxx_to_np_type : cxx_to_np_type_impl<std::remove_cv_t<T>> {};

/**
 * @brief True when T is a std::complex instantiation.
 *
 * @tparam T  A type to inspect.
 */
template <typename T> struct is_complex : std::false_type {};
template <typename T> struct is_complex<std::complex<T>> : std::true_type {};
template <typename T> inline constexpr bool is_complex_v = is_complex<T>::value;
} // namespace detail

namespace _Np_dtype {
/** @brief Trivial branch used when the other dtype branch is unused. */
struct _Np_unused_branch {};

/**
 * @brief Storage type for the character dtypes.
 *
 * @tparam _DtypeElement  A np::dtype value (`string_` or `unicode_`).
 */
template <dtype _DtypeElement> struct _Np_string_value {
  using type = std::string;
};
template <> struct _Np_string_value<dtype::unicode_> {
  using type = std::u32string;
};

/** @brief Compile-time check: the dtype stores its value as text. */
template <dtype D>
inline constexpr bool is_string_dtype_v =
    D == dtype::string_ || D == dtype::unicode_;

/**
 * @brief Union-backed storage with two branch structures.
 *
 * The first branch holds the contiguous scalar value for the
 * integral/numeric dtypes (via `value_type`); the second branch holds
 * the string attribute for `string_` / `unicode_`. Exactly one branch
 * is ever used — the other stays latent with a trivial placeholder —
 * so the numeric case keeps a literal type usable in constant
 * expressions.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 * @tparam _IsString      True when the dtype stores text.
 */
template <auto _DtypeElement, bool _IsString = is_string_dtype_v<_DtypeElement>>
struct _Np_StorageClassifier;

/**
 * @brief Integral/numeric branch of the storage classifier.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 */
template <auto _DtypeElement>
struct _Np_StorageClassifier<_DtypeElement, /* _IsString */ false> {
  using value_type = typename detail::_Np_type_to_cxx<_DtypeElement>::type;

  static constexpr np::dtype type = _DtypeElement;
  static constexpr bool is_text = false;

private:
  union _Np_storage {
    value_type value;         // numeric branch
    _Np_unused_branch unused; // string branch placeholder
  } storage_{};

public:
  constexpr _Np_StorageClassifier() noexcept = default;
  constexpr _Np_StorageClassifier(const value_type &__v)
      : storage_{.value = __v} {}
  constexpr _Np_StorageClassifier(value_type &&__v) noexcept
      : storage_{.value = static_cast<value_type &&>(__v)} {}

  constexpr auto operator=(const value_type &other) -> _Np_StorageClassifier & {
    storage_.value = other;
    return *this;
  }
  constexpr auto operator=(value_type &&other) -> _Np_StorageClassifier & {
    storage_.value = static_cast<value_type &&>(other);
    return *this;
  }

  constexpr operator value_type &() noexcept { return storage_.value; }
  constexpr operator value_type() const noexcept { return storage_.value; }

  /** @brief The compile-time dtype. */
  static constexpr auto get_type() noexcept -> np::dtype { return type; }
  /** @brief Access the underlying numeric value by reference. */
  constexpr auto value() noexcept -> value_type & { return storage_.value; }
  constexpr auto value() const noexcept -> const value_type & {
    return storage_.value;
  }
};

/**
 * @brief String branch of the storage classifier.
 *
 * @tparam _DtypeElement  A np::dtype enumeration value.
 */
template <auto _DtypeElement>
struct _Np_StorageClassifier<_DtypeElement, true> {
  using value_type = typename _Np_string_value<_DtypeElement>::type;

  static constexpr np::dtype type = _DtypeElement;
  static constexpr bool is_text = true;

private:
  union _Np_storage {
    _Np_unused_branch unused; // numeric branch placeholder
    value_type value;         // string branch

    constexpr _Np_storage() noexcept : value{} {}
    _Np_storage(const value_type &__v) noexcept : value(__v) {}
    _Np_storage(value_type &&__v) noexcept
        : value(static_cast<value_type &&>(__v)) {}

    ~_Np_storage() { value.~value_type(); }
  };

  _Np_storage storage_{};

public:
  _Np_StorageClassifier() noexcept = default;
  _Np_StorageClassifier(const value_type &__v) noexcept : storage_(__v) {}
  _Np_StorageClassifier(value_type &&__v) noexcept
      : storage_(static_cast<value_type &&>(__v)) {}
  _Np_StorageClassifier(const _Np_StorageClassifier &other) noexcept
      : storage_(other.storage_.value) {}
  _Np_StorageClassifier(_Np_StorageClassifier &&other) noexcept
      : storage_(static_cast<value_type &&>(other.storage_.value)) {}

  auto operator=(const value_type &other) -> _Np_StorageClassifier & {
    storage_.value = other;
    return *this;
  }
  auto operator=(value_type &&other) -> _Np_StorageClassifier & {
    storage_.value = static_cast<value_type &&>(other);
    return *this;
  }
  auto operator=(const _Np_StorageClassifier &other)
      -> _Np_StorageClassifier & {
    storage_.value = other.storage_.value;
    return *this;
  }
  auto operator=(_Np_StorageClassifier &&other) -> _Np_StorageClassifier & {
    storage_.value = static_cast<value_type &&>(other.storage_.value);
    return *this;
  }

  operator value_type &() noexcept { return storage_.value; }
  operator const value_type &() const noexcept { return storage_.value; }

  /** @brief The compile-time dtype. */
  static constexpr auto get_type() noexcept -> np::dtype { return type; }
  /** @brief Access the underlying string by reference. */
  auto value() noexcept -> value_type & { return storage_.value; }
  auto value() const noexcept -> const value_type & { return storage_.value; }
};

/** @brief Compile-time compare of two storage classifiers. */
template <auto _L, bool _Lb, auto _R, bool _Rb>
constexpr auto operator==(_Np_StorageClassifier<_L, _Lb>,
                          _Np_StorageClassifier<_R, _Rb>) noexcept -> bool {
  return _L == _R;
}
template <auto _L, bool _Lb, auto _R, bool _Rb>
constexpr auto operator!=(_Np_StorageClassifier<_L, _Lb>,
                          _Np_StorageClassifier<_R, _Rb>) noexcept -> bool {
  return _L != _R;
}
#ifdef _NP_USE_DIRECT_STD_TYPES
/**
 * @brief Storage type aliases mirroring the numpy C-API `npy_*`
 *        typedefs. Each alias names the native C++ storage of the
 *        matching np::dtype (the `value_type` of
 *        detail::_Np_type_to_cxx), so it can be used as a
 *        compile-time dtype tag. `string_` / `unicode_` use a string
 *        attribute; only `void_` and `object_` have no fixed
 *        C++ storage and are omitted.
 */
using _Np_int8 = std::int8_t;
using _Np_int16 = std::int16_t;
using _Np_int32 = std::int32_t;
using _Np_int64 = std::int64_t;
using _Np_uint8 = std::uint8_t;
using _Np_uint16 = std::uint16_t;
using _Np_uint32 = std::uint32_t;
using _Np_uint64 = std::uint64_t;
using _Np_float16 = std::uint16_t; // half-precision bit storage
using _Np_float32 = float;
using _Np_float64 = double;
using _Np_longdouble = long double;
using _Np_complex64 = std::complex<float>;
using _Np_complex128 = std::complex<double>;
using _Np_clongdouble = std::complex<long double>;
using _Np_bool_ = bool;
// datetime64 / timedelta64 count their units in int64_t.
using _Np_datetime64 = std::int64_t;
using _Np_timedelta64 = std::int64_t;
// String dtypes have no contiguous scalar; use a string attribute.
using _Np_string = std::string;
using _Np_unicode = std::u32string;
#else
/**
 * @brief Classifier-based storage aliases used when
 *        `_NP_USE_DIRECT_STD_TYPES` is not defined.
 *
 * Each alias is a `_Np_StorageClassifier` instantiation, binding a
 * compile-time `np::dtype` (`type`) to an instance of its native
 * C++ storage (`value_type`), so the storage is self-describing at
 * compile time while remaining usable as a plain scalar. `string_`
 * and `unicode_` use the string branch of the classifier; only
 * `void_` and `object_` have no fixed C++ storage.
 */
using _Np_int8 = _Np_StorageClassifier<np::dtype::int8>;
using _Np_int16 = _Np_StorageClassifier<np::dtype::int16>;
using _Np_int32 = _Np_StorageClassifier<np::dtype::int32>;
using _Np_int64 = _Np_StorageClassifier<np::dtype::int64>;
using _Np_uint8 = _Np_StorageClassifier<np::dtype::uint8>;
using _Np_uint16 = _Np_StorageClassifier<np::dtype::uint16>;
using _Np_uint32 = _Np_StorageClassifier<np::dtype::uint32>;
using _Np_uint64 = _Np_StorageClassifier<np::dtype::uint64>;
using _Np_float16 = _Np_StorageClassifier<np::dtype::float16>;
using _Np_float32 = _Np_StorageClassifier<np::dtype::float32>;
using _Np_float64 = _Np_StorageClassifier<np::dtype::float64>;
using _Np_longdouble = _Np_StorageClassifier<np::dtype::longdouble>;
using _Np_complex64 = _Np_StorageClassifier<np::dtype::complex64>;
using _Np_complex128 = _Np_StorageClassifier<np::dtype::complex128>;
using _Np_clongdouble = _Np_StorageClassifier<np::dtype::clongdouble>;
using _Np_bool_ = _Np_StorageClassifier<np::dtype::bool_>;
// datetime64 / timedelta64 count their units in int64_t.
using _Np_datetime64 = _Np_StorageClassifier<np::dtype::datetime64>;
using _Np_timedelta64 = _Np_StorageClassifier<np::dtype::timedelta64>;
// String dtypes use the string branch of the classifier.
using _Np_string = _Np_StorageClassifier<np::dtype::string_>;
using _Np_unicode = _Np_StorageClassifier<np::dtype::unicode_>;
#endif
} // namespace _Np_dtype

/**
 * @brief Native C++ type corresponding to a np::dtype value.
 *
 * @tparam D  A np::dtype enumeration value.
 */
template <dtype D> using dtype_t = typename detail::_Np_type_to_cxx<D>::type;

/**
 * @brief np::dtype value corresponding to a native C++ type.
 *
 * @tparam T  A native C++ type.
 */
template <typename T>
inline constexpr dtype dtype_of =
    detail::cxx_to_np_type<std::remove_cv_t<T>>::value;

/** @brief True when T is a std::complex instantiation. */
template <typename T>
inline constexpr bool is_complex_v = detail::is_complex<T>::value;

namespace detail {
/**
 * @brief Compile-time check: the dtype is one of the numeric dtypes.
 *
 * True for every dtype that has a scalar `value_type` in
 * `_Np_type_to_cxx` (integers, floats, complex, bool_, datetime64
 * and timedelta64). False for `string_`, `unicode_`, `void_` and
 * `object_`.
 *
 * @tparam D  A np::dtype enumeration value.
 */
template <dtype D>
struct is_numeric_dtype
    : std::bool_constant<D >= dtype::int8 && D != dtype::void_ &&
                         D != dtype::object_ && D != dtype::string_ &&
                         D != dtype::unicode_> {};

template <dtype D>
inline constexpr bool is_numeric_dtype_v = is_numeric_dtype<D>::value;
} // namespace detail

/**
 * @brief Compile-time check: the dtype is an integral dtype.
 *
 * Analogous to `dtype_is_integer` but evaluated at compile time from a
 * dtype value. True only for int8 through uint64; bool_ is excluded.
 *
 * @tparam D  A dtype value.
 */
template <dtype D>
using is_integral_dtype =
    std::bool_constant<(D >= dtype::int8 && D <= dtype::uint64)>;

template <dtype D>
inline constexpr bool is_integral_dtype_v = is_integral_dtype<D>::value;

/**
 * @brief Compile-time check: the dtype is numeric (non-text).
 *
 * @tparam D  A np::dtype enumeration value.
 */
template <dtype D>
inline constexpr bool is_numeric_dtype_v = detail::is_numeric_dtype<D>::value;

// Runtime helpers
/**
 * @brief Human-readable name of a dtype.
 *
 * @param t  The dtype value.
 * @return   A string_view naming the dtype (e.g. "int8", "float64").
 */
NP_NODISCARD constexpr std::string_view dtype_name(dtype t) {
  switch (t) {
  case dtype::int8:
    return "int8";
  case dtype::int16:
    return "int16";
  case dtype::int32:
    return "int32";
  case dtype::int64:
    return "int64";
  case dtype::uint8:
    return "uint8";
  case dtype::uint16:
    return "uint16";
  case dtype::uint32:
    return "uint32";
  case dtype::uint64:
    return "uint64";
  case dtype::float16:
    return "float16";
  case dtype::float32:
    return "float32";
  case dtype::float64:
    return "float64";
  case dtype::longdouble:
    return "longdouble";
  case dtype::complex64:
    return "complex64";
  case dtype::complex128:
    return "complex128";
  case dtype::clongdouble:
    return "clongdouble";
  case dtype::bool_:
    return "bool";
  case dtype::string_:
    return "str";
  case dtype::unicode_:
    return "unicode";
  case dtype::datetime64:
    return "datetime64";
  case dtype::timedelta64:
    return "timedelta64";
  case dtype::void_:
    return "void";
  case dtype::object_:
    return "object";
  }
  return "unknown";
}

/**
 * @brief Size in bytes of a dtype.
 *
 * Returns 0 for the special/variable dtypes (string_, unicode_,
 * void_, object_) and for longdouble/clongdouble (which are
 * platform-dependent).
 *
 * @param t  The dtype value.
 * @return   Number of bytes, or 0 for variable-length types.
 */
NP_NODISCARD constexpr std::size_t dtype_size(dtype t) {
  switch (t) {
  case dtype::int8:
  case dtype::uint8:
  case dtype::bool_:
    return 1;
  case dtype::int16:
  case dtype::uint16:
  case dtype::float16:
    return 2;
  case dtype::int32:
  case dtype::uint32:
  case dtype::float32:
    return 4;
  case dtype::int64:
  case dtype::uint64:
  case dtype::float64:
  case dtype::complex64:
  case dtype::datetime64:
  case dtype::timedelta64:
    return 8;
  case dtype::complex128:
    return 16;
  case dtype::longdouble:
    return sizeof(long double);
  case dtype::clongdouble:
    return sizeof(std::complex<long double>);
  case dtype::string_:
  case dtype::unicode_:
  case dtype::void_:
  case dtype::object_:
    return 0;
  }
  return 0;
}

/**
 * @brief True for the complex dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is complex64, complex128, or clongdouble.
 */
NP_NODISCARD constexpr bool dtype_is_complex(dtype t) {
  return t == dtype::complex64 || t == dtype::complex128 ||
         t == dtype::clongdouble;
}

/**
 * @brief True for floating-point (non-complex) dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is float16, float32, float64, or longdouble.
 */
NP_NODISCARD constexpr bool dtype_is_floating(dtype t) {
  return t == dtype::float16 || t == dtype::float32 || t == dtype::float64 ||
         t == dtype::longdouble;
}

/**
 * @brief True for the integer dtypes (signed or unsigned).
 *
 * @param t  The dtype value.
 * @return   True if t is int8 through uint64.
 */
NP_NODISCARD constexpr bool dtype_is_integer(dtype t) {
  return t >= dtype::int8 && t <= dtype::uint64;
}

/**
 * @brief True for signed integer dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is int8 through int64.
 */
NP_NODISCARD constexpr bool dtype_is_signed(dtype t) {
  return t >= dtype::int8 && t <= dtype::int64;
}

/**
 * @brief True for unsigned integer dtypes.
 *
 * @param t  The dtype value.
 * @return   True if t is uint8 through uint64.
 */
NP_NODISCARD constexpr bool dtype_is_unsigned(dtype t) {
  return t >= dtype::uint8 && t <= dtype::uint64;
}

/**
 * @brief True for boolean dtype.
 *
 * @param t  The dtype value.
 * @return   True if t is bool_.
 */
NP_NODISCARD constexpr bool dtype_is_bool(dtype t) {
  return t == dtype::bool_;
}

} // namespace np

#endif // NP_DTYPE_HPP
