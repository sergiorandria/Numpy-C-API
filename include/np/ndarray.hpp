/**
 * @file ndarray.hpp
 * @brief The np::Ndarray class -- a NumPy-compatible multidimensional array.
 *
 * Features:
 *  - N-dimensional storage with C-order (row-major) strides
 *  - Chained subscript access via stack-based proxies (arr[i][j][k])
 *  - Views (transpose, swapaxes, squeeze, reshape) that share storage
 *  - Reductions with optional axis (sum, mean, var, std, min, max, all, any)
 *  - Sorting / indexing helpers (sort, argsort, searchsorted, take, put)
 *  - Element-wise arithmetic with NumPy-style broadcasting
 *  - Logical iterators that honor strides (correct for views)
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_NDARRAY_HPP
#define NP_NDARRAY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dtype.hpp"
#include "exceptions.hpp"
#include "detail/proxy.hpp"

namespace np {

    namespace matrix {
        /**
         * @brief Memory layout order.
         */
        enum class Order : std::uint8_t {
            C,  ///< Row-major (C style)
            F   ///< Column-major (Fortran style)
        };
    } // namespace matrix

    namespace detail {

        /**
         * @brief True when U is a std::initializer_list instantiation.
         */
        template <typename U>
        struct is_init_list : std::false_type {};

        template <typename V>
        struct is_init_list<std::initializer_list<V>> : std::true_type {};

    } // namespace detail

    /**
     * @brief Result type of mean/var/std reductions.
     *
     * Floating and complex inputs keep their type; integer and boolean
     * inputs promote to double (NumPy semantics).
     */
    template <typename T>
    struct _mean_type {
        using type = std::conditional_t<std::is_floating_point_v<T> ||
                                            detail::is_complex_v<T>,
                                        T, double>;
    };

    template <typename T>
    class Matrix;

    // ---------------------------------------------------------------------
    // Logical iterator (stride-aware, correct for views)
    // ---------------------------------------------------------------------

    /**
     * @brief Forward iterator visiting array elements in logical (C) order.
     * @tparam T Element type; instantiate with const T for read-only access.
     */
    template <typename T>
    class ndarray_iterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = std::remove_const_t<T>;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = T&;

        ndarray_iterator(T* base, std::vector<std::size_t> shape,
                         std::vector<std::size_t> strides, bool at_end)
            : base_(base),
              shape_(std::move(shape)),
              strides_(std::move(strides)),
              idx_(shape_.size(), 0),
              done_(at_end) {}

        [[nodiscard]] reference operator*() const {
            return base_[detail::flat_index(idx_, strides_, 0)];
        }

        [[nodiscard]] pointer operator->() const {
            return &base_[detail::flat_index(idx_, strides_, 0)];
        }

        ndarray_iterator& operator++() {
            _advance();
            return *this;
        }

        ndarray_iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        [[nodiscard]] bool operator==(const ndarray_iterator& o) const noexcept {
            if (base_ != o.base_ || done_ != o.done_) {
                return false;
            }
            return done_ || idx_ == o.idx_;
        }

        [[nodiscard]] bool operator!=(const ndarray_iterator& o) const noexcept {
            return !(*this == o);
        }

      private:
        void _advance() noexcept {
            if (shape_.empty()) {
                done_ = true;
                return;
            }
            for (std::size_t d = shape_.size(); d-- > 0;) {
                if (++idx_[d] < shape_[d]) {
                    return;
                }
                idx_[d] = 0;
            }
            done_ = true;
        }

        T* base_;
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;
        std::vector<std::size_t> idx_;
        bool done_;
    };

    // ---------------------------------------------------------------------
    // Ndarray
    // ---------------------------------------------------------------------

    /**
     * @brief A NumPy-style multidimensional array container.
     *
     * @tparam T Element type (numeric or std::complex).
     */
    template <typename T = double>
    class Ndarray {
      public:
        using value_type       = T;
        using size_type        = std::size_t;
        using iterator         = ndarray_iterator<T>;
        using const_iterator   = ndarray_iterator<const T>;
        /**
         * @brief Reference type returned by non-const element accessors.
         *
         * std::vector<bool> is specialized, so its element access yields a
         * proxy type rather than bool&; this alias keeps the Ndarray API
         * uniform for bool arrays.
         */
        using reference = std::conditional_t<
            std::is_same_v<T, bool>, std::vector<bool>::reference, T&>;

        // --- attributes (mirror ndarray.shape / strides / dtype / order) ---
        std::vector<int> shape;         ///< Dimensions of the array
        std::vector<std::size_t> strides; ///< Strides in elements
        np::dtype type = dtype::void_;  ///< Data type
        matrix::Order order = matrix::Order::C;  ///< Memory layout
        std::size_t offset = 0;         ///< Element offset into storage (views)

        // =================================================================
        // Construction
        // =================================================================

        /** @brief Default constructor: empty 0-dimensional array. */
        Ndarray() = default;

        /**
         * @brief Constructs an array of the given shape, filled with `fill`.
         */
        explicit Ndarray(const std::vector<int>& shape,
                         np::dtype type = dtype::void_,
                         const T& fill = T{});

        /**
         * @brief Builds an array from an owned data buffer.
         *
         * Provided as a static factory (instead of a constructor) so that
         * nested-brace construction like `Ndarray<int> a{{1,2},{3,4}}`
         * unambiguously selects the nested initializer-list constructor.
         *
         * @throws std::invalid_argument if data.size() != product(shape).
         */
        static Ndarray from_data(const std::vector<int>& shape,
                                 std::vector<T> data);

        /** @brief 1D construction from a flat initializer list. */
        Ndarray(std::initializer_list<T> list);

        /**
         * @brief 2D construction from nested initializer lists, e.g.
         *        `Ndarray<int> a{{1, 2}, {3, 4}}`.
         * @throws std::invalid_argument on ragged (inconsistent) rows.
         */
        template <typename U>
        Ndarray(std::initializer_list<std::initializer_list<U>> rows);

        /**
         * @brief Deep-copying copy constructor (value semantics).
         */
        Ndarray(const Ndarray& other);

        /** @brief Move constructor: transfers storage. */
        Ndarray(Ndarray&&) noexcept = default;

        /** @brief Deep-copying copy assignment (value semantics). */
        Ndarray& operator=(const Ndarray& other);

        /** @brief Move assignment: transfers storage. */
        Ndarray& operator=(Ndarray&&) noexcept = default;

        // =================================================================
        // Attributes
        // =================================================================

        /** @brief Total number of elements. */
        [[nodiscard]] std::size_t size() const noexcept;

        /** @brief Number of dimensions. */
        [[nodiscard]] std::size_t ndim() const noexcept;

        /** @brief Bytes per element. */
        [[nodiscard]] std::size_t itemsize() const noexcept;

        /** @brief Total bytes consumed by the logical elements. */
        [[nodiscard]] std::size_t nbytes() const noexcept;

        /** @brief True if the array has no elements. */
        [[nodiscard]] bool empty() const noexcept;

        /** @brief True when the logical elements are laid out contiguously. */
        [[nodiscard]] bool is_contiguous() const noexcept;

        /** @brief Writable access to the underlying storage buffer. */
        std::vector<T>& data();

        /** @brief Read-only access to the underlying storage buffer. */
        [[nodiscard]] const std::vector<T>& data() const;

        /** @brief Product of the shape (total element count). */
        [[nodiscard]] std::size_t _numel() const noexcept;

        /** @brief Flat logical offset of a multi-index. */
        [[nodiscard]] std::size_t
        _flat(const std::vector<std::size_t>& idx) const noexcept;

        /** @brief Physical storage offset of flat logical position i. */
        [[nodiscard]] std::size_t _flat_logical(std::size_t i) const noexcept;

        // =================================================================
        // Iterators
        // =================================================================

        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }

        // =================================================================
        // Element access
        // =================================================================

        /**
         * @brief Chained subscript access (read/write).
         */
        auto operator[](std::size_t index) -> Proxy<T>;

        /** @brief Chained subscript access (read-only). */
        auto operator[](std::size_t index) const -> ConstProxy<T>;

        /** @brief Compile-time-size index access (reference). */
        template <std::size_t N>
        auto get(const std::array<std::size_t, N>& idx) -> reference;

        /** @brief Compile-time-size index access (const reference). */
        template <std::size_t N>
        auto get(const std::array<std::size_t, N>& idx) const -> const T&;

        /** @brief Runtime index container access (by value). */
        template <typename Container>
        auto get(const Container& idx) const -> T;

        /** @brief Write a value at runtime index container position. */
        template <typename Container>
        void set(const Container& idx, const T& value);

        /** @brief 1D bounds-checked access. */
        auto at(std::size_t i) -> reference;

        /** @brief 1D bounds-checked access (const). */
        auto at(std::size_t i) const -> const T&;

        /** @brief Single-index access for 1D arrays (read/write). */
        auto operator()(std::size_t i) -> reference;

        /** @brief Single-index access for 1D arrays (const). */
        auto operator()(std::size_t i) const -> const T&;

        /** @brief 2D index access (read/write). */
        auto operator()(std::size_t i, std::size_t j) -> reference;

        /** @brief 2D index access (const). */
        auto operator()(std::size_t i, std::size_t j) const -> const T&;

        /** @brief 2D bounds-checked access. */
        auto at(std::size_t i, std::size_t j) -> reference;

        /** @brief 2D bounds-checked access (const). */
        auto at(std::size_t i, std::size_t j) const -> const T&;

        /** @brief Returns the single element of a 0-d/1-element array. */
        T item() const;

        // =================================================================
        // Reductions
        // =================================================================

        /** @brief Sum over all elements. */
        auto sum() const -> std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        /** @brief Sum along an axis. */
        template <typename Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>>
        auto sum(int axis, bool keepdims = false) const -> Ndarray<Acc>;

        /** @brief Product over all elements. */
        auto prod() const -> std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        /** @brief Product along an axis. */
        template <typename Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>>
        auto prod(int axis, bool keepdims = false) const -> Ndarray<Acc>;

        /** @brief Minimum over all elements. */
        T min() const;
        /** @brief Minimum along an axis. */
        auto min(int axis, bool keepdims = false) const -> Ndarray<T>;

        /** @brief Maximum over all elements. */
        T max() const;
        /** @brief Maximum along an axis. */
        auto max(int axis, bool keepdims = false) const -> Ndarray<T>;

        /** @brief Arithmetic mean over all elements. */
        auto mean() const -> typename _mean_type<T>::type;
        /** @brief Arithmetic mean along an axis. */
        auto mean(int axis, bool keepdims = false) const
            -> Ndarray<typename _mean_type<T>::type>;

        /** @brief Population variance over all elements. */
        auto var() const -> typename _mean_type<T>::type;
        /** @brief Population variance along an axis. */
        auto var(int axis, bool keepdims = false) const
            -> Ndarray<typename _mean_type<T>::type>;

        /** @brief Population standard deviation over all elements. */
        auto std() const -> typename _mean_type<T>::type;
        /** @brief Population standard deviation along an axis. */
        auto std(int axis, bool keepdims = false) const
            -> Ndarray<typename _mean_type<T>::type>;

        /** @brief True when every element is non-zero. */
        bool all() const;
        /** @brief All along an axis. */
        auto all(int axis, bool keepdims = false) const -> Ndarray<bool>;

        /** @brief True when any element is non-zero. */
        bool any() const;
        /** @brief Any along an axis. */
        auto any(int axis, bool keepdims = false) const -> Ndarray<bool>;

        /** @brief Flat logical index of the maximum element. */
        std::size_t argmax() const;
        /** @brief Indices of maxima along an axis. */
        auto argmax(int axis, bool keepdims = false) const -> Ndarray<std::size_t>;

        /** @brief Flat logical index of the minimum element. */
        std::size_t argmin() const;
        /** @brief Indices of minima along an axis. */
        auto argmin(int axis, bool keepdims = false) const -> Ndarray<std::size_t>;

        /** @brief Cumulative sum (flattened when no axis is given). */
        auto cumsum() const
            -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>>;
        /** @brief Cumulative sum along an axis. */
        auto cumsum(int axis) const
            -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>>;

        /** @brief Cumulative product (flattened when no axis is given). */
        auto cumprod() const
            -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>>;
        /** @brief Cumulative product along an axis. */
        auto cumprod(int axis) const
            -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>>;

        // =================================================================
        // Sorting / searching
        // =================================================================

        /** @brief In-place sort along an axis (default: last axis). */
        void sort(int axis = -1);

        /** @brief Sorted copy of the array along an axis (default: last). */
        auto sorted(int axis = -1) const -> Ndarray<T>;

        /** @brief Indices that would sort the array along an axis. */
        auto argsort(int axis = -1) const -> Ndarray<std::size_t>;

        /** @brief Indices that would partition at position k along an axis. */
        auto argpartition(std::size_t kth, int axis = -1) const
            -> Ndarray<std::size_t>;

        /** @brief Binary search for a value in a sorted 1D array. */
        std::size_t searchsorted(const T& value,
                                 bool side_right = false) const;

        /** @brief Searchsorted applied to every element of `values`. */
        auto searchsorted(const Ndarray<int>& values) const
            -> Ndarray<std::size_t>;

        // =================================================================
        // Shape manipulation
        // =================================================================

        /**
         * @brief View (when contiguous) or copy with a new shape.
         * @param shape New shape; at most one dimension may be -1.
         */
        auto reshape(const std::vector<int>& shape) const -> Ndarray;

        /** @brief View with reversed dimensions. */
        auto transpose() const -> Ndarray;

        /** @brief View with a permutation of the dimensions. */
        auto transpose(const std::vector<int>& perm) const -> Ndarray;

        /** @brief View with two axes swapped. */
        auto swapaxes(int axis1, int axis2) const -> Ndarray;

        /** @brief View removing all size-1 dimensions. */
        auto squeeze() const -> Ndarray;

        /** @brief View removing a specific dimension. */
        auto squeeze(int axis) const -> Ndarray;

        /** @brief View (contiguous) or copy flattened in C order. */
        auto ravel() const -> Ndarray;

        /** @brief Copy flattened in C order. */
        auto flatten() const -> Ndarray;

        /** @brief Resize in place to a new total number of elements. */
        void resize(const std::vector<int>& new_shape);

        // =================================================================
        // Manipulation
        // =================================================================

        /** @brief Fill every element with a value. */
        void fill(const T& value);

        /** @brief Deep copy of the array. */
        auto copy() const -> Ndarray;

        /** @brief View sharing the same storage. */
        auto view() const -> Ndarray;

        /** @brief Element-wise conversion to another type. */
        template <typename U>
        auto astype() const -> Ndarray<U>;

        /** @brief Gather elements along an axis (default: flattened). */
        auto take(const std::vector<std::size_t>& indices, int axis = 0) const
            -> Ndarray;

        /** @brief Set elements at flat logical positions. */
        void put(const std::vector<std::size_t>& indices,
                 const std::vector<T>& values, char mode = 'r');

        /** @brief Repeat elements (flattened when no axis given). */
        auto repeat(std::size_t repeats) const -> Ndarray;
        auto repeat(std::size_t repeats, int axis) const -> Ndarray;

        /** @brief Clip values into [min_value, max_value]. */
        auto clip(const T& min_value, const T& max_value) const -> Ndarray;

        /** @brief Round to `decimals` places. */
        auto round(int decimals = 0) const -> Ndarray;

        /** @brief Diagonal of a 2D+ array. */
        auto diagonal(int offset = 0) const -> Ndarray;

        /** @brief Sum along the diagonal. */
        T trace(int offset = 0) const;

        /** @brief Indices of non-zero elements (one array per dimension). */
        auto nonzero() const -> std::vector<Ndarray<std::size_t>>;

        /** @brief Element-wise complex conjugate. */
        auto conj() const -> Ndarray;

        /** @brief Swap the byte order of every element, in place. */
        void byteswap();

        // =================================================================
        // Conversions / IO
        // =================================================================

        /** @brief Flat logical elements as a std::vector. */
        auto tolist() const -> std::vector<T>;

        /** @brief Native-endian byte dump of the logical elements. */
        auto tobytes() const -> std::vector<std::uint8_t>;

        /** @brief Write the raw bytes to a binary file. */
        void tofile(const std::string& filename) const;

        /** @brief Write the raw bytes to an output stream. */
        void tofile(std::ostream& os) const;

        /** @brief Human-readable representation. */
        void print(std::ostream& os = std::cout) const;

        // =================================================================
        // Element-wise arithmetic (broadcasting)
        // =================================================================

        template <typename U>
        auto operator+(const Ndarray<U>& rhs) const
            -> Ndarray<std::common_type_t<T, U>>;
        template <typename U>
        auto operator-(const Ndarray<U>& rhs) const
            -> Ndarray<std::common_type_t<T, U>>;
        template <typename U>
        auto operator*(const Ndarray<U>& rhs) const
            -> Ndarray<std::common_type_t<T, U>>;
        template <typename U>
        auto operator/(const Ndarray<U>& rhs) const
            -> Ndarray<std::common_type_t<T, U>>;

        template <typename U>
        auto operator+(const U& scalar) const
            -> Ndarray<std::common_type_t<T, U>>;
        template <typename U>
        auto operator-(const U& scalar) const
            -> Ndarray<std::common_type_t<T, U>>;
        template <typename U>
        auto operator*(const U& scalar) const
            -> Ndarray<std::common_type_t<T, U>>;
        template <typename U>
        auto operator/(const U& scalar) const
            -> Ndarray<std::common_type_t<T, U>>;

        /** @brief Unary negation (element-wise). */
        auto operator-() const -> Ndarray;

        // Comparisons (element-wise, NumPy semantics)
        template <typename U>
        auto operator==(const Ndarray<U>& rhs) const -> Ndarray<bool>;
        template <typename U>
        auto operator!=(const Ndarray<U>& rhs) const -> Ndarray<bool>;
        template <typename U>
        auto operator<(const Ndarray<U>& rhs) const -> Ndarray<bool>;
        template <typename U>
        auto operator<=(const Ndarray<U>& rhs) const -> Ndarray<bool>;
        template <typename U>
        auto operator>(const Ndarray<U>& rhs) const -> Ndarray<bool>;
        template <typename U>
        auto operator>=(const Ndarray<U>& rhs) const -> Ndarray<bool>;

        template <typename U>
        auto operator==(const U& scalar) const -> Ndarray<bool>;
        template <typename U>
        auto operator!=(const U& scalar) const -> Ndarray<bool>;
        template <typename U>
        auto operator<(const U& scalar) const -> Ndarray<bool>;
        template <typename U>
        auto operator<=(const U& scalar) const -> Ndarray<bool>;
        template <typename U>
        auto operator>(const U& scalar) const -> Ndarray<bool>;
        template <typename U>
        auto operator>=(const U& scalar) const -> Ndarray<bool>;

        /** @brief True if same shape and all elements equal. */
        bool all_equal(const Ndarray& other) const noexcept;

        /** @brief True if all elements equal the given value. */
        bool all_equal(const T& value) const noexcept;

        // In-place arithmetic (same shape, or broadcast for += etc.)
        Ndarray& operator+=(const Ndarray& rhs);
        Ndarray& operator-=(const Ndarray& rhs);
        Ndarray& operator*=(const Ndarray& rhs);
        Ndarray& operator/=(const Ndarray& rhs);
        Ndarray& operator+=(const T& scalar);
        Ndarray& operator-=(const T& scalar);
        Ndarray& operator*=(const T& scalar);
        Ndarray& operator/=(const T& scalar);

        // Scalar-on-the-left friends
        template <typename U>
        friend auto operator+(const U& scalar, const Ndarray& arr)
            -> Ndarray<std::common_type_t<U, T>> {
            return arr + scalar;
        }
        template <typename U>
        friend auto operator-(const U& scalar, const Ndarray& arr)
            -> Ndarray<std::common_type_t<U, T>> {
            return arr._scalar_left_op(scalar,
                                       [](const U& a, const T& b) { return a - b; });
        }
        template <typename U>
        friend auto operator*(const U& scalar, const Ndarray& arr)
            -> Ndarray<std::common_type_t<U, T>> {
            return arr._scalar_left_op(scalar,
                                       [](const U& a, const T& b) { return a * b; });
        }
        template <typename U>
        friend auto operator/(const U& scalar, const Ndarray& arr)
            -> Ndarray<std::common_type_t<U, T>> {
            return arr._scalar_left_op(scalar,
                                       [](const U& a, const T& b) { return a / b; });
        }

        /** @brief Stream output in NumPy repr style. */
        friend auto operator<<(std::ostream& os, const Ndarray& arr)
            -> std::ostream& {
            arr._print_to(os);
            return os;
        }

      private:
        // -----------------------------------------------------------------
        // Internals
        // -----------------------------------------------------------------

        template <typename U>
        friend class Ndarray;

        std::shared_ptr<std::vector<T>> data_;  ///< Shared storage (enables views)

        /** @brief View constructor (shares storage). */
        Ndarray(std::shared_ptr<std::vector<T>> data, std::vector<int> shape,
                std::vector<std::size_t> strides, np::dtype type,
                matrix::Order order, std::size_t offset);

        /** @brief C-order strides for a shape. */
        [[nodiscard]] static std::vector<std::size_t>
        _c_strides(const std::vector<int>& shape) noexcept;

        /** @brief shape as std::size_t vector. */
        [[nodiscard]] std::vector<std::size_t> _shape_u() const noexcept;

        /** @brief Normalize a possibly negative axis. */
        [[nodiscard]] int _normalize_axis(int axis) const;

        /** @brief Visit every logical element. */
        template <typename Fn>
        void _for_each_logical(Fn&& fn) const;

        /** @brief Visit every logical element with its multi-index. */
        template <typename Fn>
        void _for_each_indexed(Fn&& fn) const;

        /** @brief Generic axis reduction. */
        template <typename Acc, typename StepFn>
        auto _reduce_axis(int axis, bool keepdims, std::optional<Acc> seed,
                          StepFn&& step) const -> Ndarray<Acc>;

        /** @brief Welford-based variance along an axis. */
        template <typename MeanT>
        auto _var_axis(int axis, bool keepdims) const -> Ndarray<MeanT>;

        /** @brief Generic extrema/arg reduction along an axis. */
        template <typename Cmp>
        auto _arg_reduce_axis(int axis, bool keepdims, Cmp&& cmp) const
            -> Ndarray<std::size_t>;

        /** @brief Internal flat write used by cumsum/cumprod. */
        template <typename Acc, typename Fn>
        auto _cum_axis(int axis, Fn&& fn) const -> Ndarray<Acc>;

        /** @brief Scalar element-wise operation over own shape. */
        template <typename U, typename Fn>
        auto _scalar_op(const U& scalar, Fn&& fn) const
            -> Ndarray<std::common_type_t<T, U>>;

        /** @brief Scalar on the left (a op b[i] with a first). */
        template <typename U, typename Fn>
        auto _scalar_left_op(const U& scalar, Fn&& fn) const
            -> Ndarray<std::common_type_t<U, T>>;

        /** @brief Scalar comparison producing a bool array. */
        template <typename U, typename Fn>
        auto _cmp_scalar(const U& scalar, Fn&& fn) const -> Ndarray<bool>;

        /** @brief Recursive printing. */
        void _print_recursive(std::size_t dim, std::size_t flat_offset,
                              std::ostream& os) const;

        /** @brief Full repr. */
        void _print_to(std::ostream& os) const;

        /** @brief Storage pointer for iterators. */
        T* _raw_ptr() noexcept;
        const T* _raw_ptr() const noexcept;

        /** @brief Finalize strides/type after construction. */
        void _finalize();

        template <typename U>
        static constexpr bool _is_valid_scalar =
            std::is_arithmetic_v<U> || detail::is_complex_v<U>;
    };

    // =====================================================================
    // Broadcasting helpers
    // =====================================================================

    namespace detail {

        /**
         * @brief NumPy-style broadcast of two shapes.
         * @throws std::invalid_argument if the shapes cannot be broadcast.
         */
        [[nodiscard]] inline std::vector<int>
        broadcast_shapes(const std::vector<int>& a, const std::vector<int>& b) {
            const int na = static_cast<int>(a.size());
            const int nb = static_cast<int>(b.size());
            const int nr = std::max(na, nb);
            std::vector<int> r(nr);
            for (int d = 0; d < nr; ++d) {
                const int ia = na - nr + d;
                const int ib = nb - nr + d;
                const int sa = ia < 0 ? 1 : a[ia];
                const int sb = ib < 0 ? 1 : b[ib];
                if (sa == sb) {
                    r[d] = sa;
                } else if (sa == 1) {
                    r[d] = sb;
                } else if (sb == 1) {
                    r[d] = sa;
                } else {
                    throw std::invalid_argument(
                        "operands could not be broadcast together");
                }
            }
            return r;
        }

        /**
         * @brief Element-wise operation with broadcasting.
         * @tparam Fn callable taking (const R&, const S&).
         */
        template <typename R, typename S, typename Fn>
        auto elementwise(const Ndarray<R>& a, const Ndarray<S>& b, Fn&& fn) {
            using OutT = std::invoke_result_t<Fn, R, S>;
            const std::vector<int> out_shape = broadcast_shapes(a.shape, b.shape);
            Ndarray<OutT> out(out_shape);

            const int nr = static_cast<int>(out_shape.size());
            const int shift_a = nr - static_cast<int>(a.shape.size());
            const int shift_b = nr - static_cast<int>(b.shape.size());

            std::vector<std::size_t> adj_a(nr), adj_b(nr);
            for (int d = 0; d < nr; ++d) {
                const int ka = d - shift_a;
                const int kb = d - shift_b;
                adj_a[d] = (ka < 0 || a.shape[ka] == 1) ? 0 : a.strides[ka];
                adj_b[d] = (kb < 0 || b.shape[kb] == 1) ? 0 : b.strides[kb];
            }

            Odometer od(out_shape);
            while (!od.done()) {
                const auto& idx = od.idx();
                std::size_t fa = a.offset, fb = b.offset, fo = 0;
                for (int d = 0; d < nr; ++d) {
                    fa += idx[d] * adj_a[d];
                    fb += idx[d] * adj_b[d];
                    fo += idx[d] * out.strides[d];
                }
                out.data()[fo] = fn(a.data()[fa], b.data()[fb]);
                od.advance();
            }
            return out;
        }

    } // namespace detail

    // =====================================================================
    // Implementation
    // =====================================================================

    template <typename T>
    Ndarray<T>::Ndarray(const std::vector<int>& shape, np::dtype type,
                        const T& fill)
        : shape(shape), type(type),
          data_(std::make_shared<std::vector<T>>(_numel(), fill)) {
        _finalize();
    }

    template <typename T>
    auto Ndarray<T>::from_data(const std::vector<int>& shape,
                               std::vector<T> data) -> Ndarray {
        Ndarray out;
        out.shape = shape;
        out.data_ = std::make_shared<std::vector<T>>(std::move(data));
        if (out.data_->size() != out._numel()) {
            throw std::invalid_argument(
                "data size does not match the array shape");
        }
        out._finalize();
        return out;
    }

    template <typename T>
    Ndarray<T>::Ndarray(std::initializer_list<T> list)
        : data_(std::make_shared<std::vector<T>>(list.begin(), list.end())) {
        shape = {static_cast<int>(list.size())};
        _finalize();
    }

    template <typename T>
    template <typename U>
    Ndarray<T>::Ndarray(std::initializer_list<std::initializer_list<U>> rows) {
        const int n_rows = static_cast<int>(rows.size());
        const int n_cols =
            n_rows > 0 ? static_cast<int>(rows.begin()->size()) : 0;
        shape = {n_rows, n_cols};
        data_ = std::make_shared<std::vector<T>>(_numel(), T{});
        std::size_t k = 0;
        for (const auto& row : rows) {
            if (static_cast<int>(row.size()) != n_cols) {
                throw std::invalid_argument(
                    "ragged rows in nested initializer list");
            }
            for (const U& v : row) {
                (*data_)[k++] = static_cast<T>(v);
            }
        }
        _finalize();
    }

    template <typename T>
    Ndarray<T>::Ndarray(const Ndarray& other)
        : shape(other.shape), strides(other.strides), type(other.type),
          order(other.order), offset(other.offset) {
        if (other.data_) {
            data_ = std::make_shared<std::vector<T>>(*other.data_);
        }
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator=(const Ndarray& other) {
        if (this != &other) {
            shape = other.shape;
            strides = other.strides;
            type = other.type;
            order = other.order;
            offset = other.offset;
            data_ = other.data_
                        ? std::make_shared<std::vector<T>>(*other.data_)
                        : nullptr;
        }
        return *this;
    }

    template <typename T>
    Ndarray<T>::Ndarray(std::shared_ptr<std::vector<T>> data,
                        std::vector<int> shape, std::vector<std::size_t> strides,
                        np::dtype type, matrix::Order order, std::size_t offset)
        : shape(std::move(shape)), strides(std::move(strides)), type(type),
          order(order), offset(offset), data_(std::move(data)) {}

    // ---------------------------------------------------------------------
    // Attributes
    // ---------------------------------------------------------------------

    template <typename T>
    auto Ndarray<T>::size() const noexcept -> std::size_t {
        return _numel();
    }

    template <typename T>
    auto Ndarray<T>::ndim() const noexcept -> std::size_t {
        return shape.size();
    }

    template <typename T>
    auto Ndarray<T>::itemsize() const noexcept -> std::size_t {
        return sizeof(T);
    }

    template <typename T>
    auto Ndarray<T>::nbytes() const noexcept -> std::size_t {
        return _numel() * sizeof(T);
    }

    template <typename T>
    bool Ndarray<T>::empty() const noexcept {
        return _numel() == 0;
    }

    template <typename T>
    bool Ndarray<T>::is_contiguous() const noexcept {
        if (strides != _c_strides(shape) || offset != 0) {
            return false;
        }
        return !data_ || data_->size() >= _numel();
    }

    template <typename T>
    auto Ndarray<T>::data() -> std::vector<T>& {
        if (!data_) {
            data_ = std::make_shared<std::vector<T>>(_numel(), T{});
        }
        return *data_;
    }

    template <typename T>
    auto Ndarray<T>::data() const -> const std::vector<T>& {
        if (!data_) {
            throw std::runtime_error("ndarray has no data buffer");
        }
        return *data_;
    }

    // ---------------------------------------------------------------------
    // Iterators
    // ---------------------------------------------------------------------

    template <typename T>
    auto Ndarray<T>::_raw_ptr() noexcept -> T* {
        return data_ ? data_->data() + offset : nullptr;
    }

    template <typename T>
    auto Ndarray<T>::_raw_ptr() const noexcept -> const T* {
        return data_ ? data_->data() + offset : nullptr;
    }

    template <typename T>
    auto Ndarray<T>::begin() -> iterator {
        return iterator(_raw_ptr(), _shape_u(), strides, _numel() == 0);
    }

    template <typename T>
    auto Ndarray<T>::end() -> iterator {
        return iterator(_raw_ptr(), _shape_u(), strides, true);
    }

    template <typename T>
    auto Ndarray<T>::begin() const -> const_iterator {
        return const_iterator(_raw_ptr(), _shape_u(), strides, _numel() == 0);
    }

    template <typename T>
    auto Ndarray<T>::end() const -> const_iterator {
        return const_iterator(_raw_ptr(), _shape_u(), strides, true);
    }

    // ---------------------------------------------------------------------
    // Element access
    // ---------------------------------------------------------------------

    template <typename T>
    auto Ndarray<T>::operator[](std::size_t index) -> Proxy<T> {
        detail::IndexStack<> idx;
        idx.push_back(index);
        return Proxy<T>(*this, idx);
    }

    template <typename T>
    auto Ndarray<T>::operator[](std::size_t index) const -> ConstProxy<T> {
        detail::IndexStack<> idx;
        idx.push_back(index);
        return ConstProxy<T>(*this, idx);
    }

    template <typename T>
    template <std::size_t N>
    auto Ndarray<T>::get(const std::array<std::size_t, N>& idx) -> reference {
        if (N != shape.size()) {
            throw std::invalid_argument(
                "index dimensionality does not match array dimensions");
        }
        std::size_t flat = offset;
        for (std::size_t i = 0; i < N; ++i) {
            if (idx[i] >= static_cast<std::size_t>(shape[i])) {
                throw std::out_of_range("index out of bounds");
            }
            flat += idx[i] * strides[i];
        }
        return (*data_)[flat];
    }

    template <typename T>
    template <std::size_t N>
    auto Ndarray<T>::get(const std::array<std::size_t, N>& idx) const
        -> const T& {
        if (N != shape.size()) {
            throw std::invalid_argument(
                "index dimensionality does not match array dimensions");
        }
        std::size_t flat = offset;
        for (std::size_t i = 0; i < N; ++i) {
            if (idx[i] >= static_cast<std::size_t>(shape[i])) {
                throw std::out_of_range("index out of bounds");
            }
            flat += idx[i] * strides[i];
        }
        return (*data_)[flat];
    }

    template <typename T>
    template <typename Container>
    auto Ndarray<T>::get(const Container& idx) const -> T {
        if (idx.size() != shape.size()) {
            throw std::invalid_argument(
                "index dimensionality does not match array dimensions");
        }
        std::size_t flat = offset;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] >= static_cast<std::size_t>(shape[i])) {
                throw std::out_of_range("index out of bounds");
            }
            flat += idx[i] * strides[i];
        }
        return (*data_)[flat];
    }

    template <typename T>
    template <typename Container>
    void Ndarray<T>::set(const Container& idx, const T& value) {
        if (idx.size() != shape.size()) {
            throw std::invalid_argument(
                "index dimensionality does not match array dimensions");
        }
        std::size_t flat = offset;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] >= static_cast<std::size_t>(shape[i])) {
                throw std::out_of_range("index out of bounds");
            }
            flat += idx[i] * strides[i];
        }
        (*data_)[flat] = value;
    }

    template <typename T>
    auto Ndarray<T>::at(std::size_t i) -> reference {
        if (shape.size() != 1) {
            throw std::invalid_argument("at() requires a 1D array");
        }
        if (i >= static_cast<std::size_t>(shape[0])) {
            throw std::out_of_range("index out of bounds");
        }
        return (*data_)[offset + i * strides[0]];
    }

    template <typename T>
    auto Ndarray<T>::at(std::size_t i) const -> const T& {
        if (shape.size() != 1) {
            throw std::invalid_argument("at() requires a 1D array");
        }
        if (i >= static_cast<std::size_t>(shape[0])) {
            throw std::out_of_range("index out of bounds");
        }
        return (*data_)[offset + i * strides[0]];
    }

    template <typename T>
    T Ndarray<T>::item() const {
        if (_numel() != 1) {
            throw std::invalid_argument(
                "can only convert an array of size 1 to a scalar");
        }
        if (!data_) {
            return T{};
        }
        return (*data_)[offset];
    }

    template <typename T>
    auto Ndarray<T>::operator()(std::size_t i) -> reference {
        if (shape.size() != 1) {
            throw std::invalid_argument(
                "operator()(i) requires a 1D array");
        }
        return (*data_)[offset + i * strides[0]];
    }

    template <typename T>
    auto Ndarray<T>::operator()(std::size_t i) const -> const T& {
        if (shape.size() != 1) {
            throw std::invalid_argument(
                "operator()(i) requires a 1D array");
        }
        return (*data_)[offset + i * strides[0]];
    }

    template <typename T>
    auto Ndarray<T>::operator()(std::size_t i, std::size_t j) -> reference {
        if (shape.size() != 2) {
            throw std::invalid_argument(
                "operator()(i, j) requires a 2D array");
        }
        return (*data_)[offset + i * strides[0] + j * strides[1]];
    }

    template <typename T>
    auto Ndarray<T>::operator()(std::size_t i, std::size_t j) const -> const T& {
        if (shape.size() != 2) {
            throw std::invalid_argument(
                "operator()(i, j) requires a 2D array");
        }
        return (*data_)[offset + i * strides[0] + j * strides[1]];
    }

    template <typename T>
    auto Ndarray<T>::at(std::size_t i, std::size_t j) -> reference {
        if (shape.size() != 2) {
            throw std::invalid_argument("at(i, j) requires a 2D array");
        }
        if (i >= static_cast<std::size_t>(shape[0]) ||
            j >= static_cast<std::size_t>(shape[1])) {
            throw std::out_of_range("index out of bounds");
        }
        return (*data_)[offset + i * strides[0] + j * strides[1]];
    }

    template <typename T>
    auto Ndarray<T>::at(std::size_t i, std::size_t j) const -> const T& {
        if (shape.size() != 2) {
            throw std::invalid_argument("at(i, j) requires a 2D array");
        }
        if (i >= static_cast<std::size_t>(shape[0]) ||
            j >= static_cast<std::size_t>(shape[1])) {
            throw std::out_of_range("index out of bounds");
        }
        return (*data_)[offset + i * strides[0] + j * strides[1]];
    }

    // ---------------------------------------------------------------------
    // Internals
    // ---------------------------------------------------------------------

    template <typename T>
    auto Ndarray<T>::_numel() const noexcept -> std::size_t {
        std::size_t n = 1;
        for (int d : shape) {
            n *= static_cast<std::size_t>(d);
        }
        return n;
    }

    template <typename T>
    auto Ndarray<T>::_c_strides(const std::vector<int>& s) noexcept
        -> std::vector<std::size_t> {
        std::vector<std::size_t> st(s.size(), 1);
        std::size_t stride = 1;
        for (std::size_t i = s.size(); i-- > 0;) {
            st[i] = stride;
            stride *= static_cast<std::size_t>(s[i]);
        }
        return st;
    }

    template <typename T>
    auto Ndarray<T>::_flat(const std::vector<std::size_t>& idx) const noexcept
        -> std::size_t {
        return detail::flat_index(idx, strides, offset);
    }

    template <typename T>
    auto Ndarray<T>::_flat_logical(std::size_t i) const noexcept
        -> std::size_t {
        if (shape.empty() || i == 0) {
            return offset;
        }
        std::vector<std::size_t> idx = _shape_u();
        std::size_t rem = i;
        for (std::size_t d = shape.size(); d-- > 0;) {
            idx[d] = rem % static_cast<std::size_t>(shape[d]);
            rem /= static_cast<std::size_t>(shape[d]);
        }
        return _flat(idx);
    }

    template <typename T>
    auto Ndarray<T>::_shape_u() const noexcept -> std::vector<std::size_t> {
        std::vector<std::size_t> u(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i) {
            u[i] = static_cast<std::size_t>(shape[i]);
        }
        return u;
    }

    template <typename T>
    auto Ndarray<T>::_normalize_axis(int axis) const -> int {
        const int nd = static_cast<int>(shape.size());
        if (axis < 0) {
            axis += nd;
        }
        if (axis < 0 || axis >= nd) {
            throw np::AxisError(
                "axis " + std::to_string(axis - (axis < 0 ? nd : 0)) +
                " is out of bounds for array of dimension " +
                std::to_string(nd));
        }
        return axis;
    }

    template <typename T>
    template <typename Fn>
    void Ndarray<T>::_for_each_logical(Fn&& fn) const {
        if (!data_) {
            return;
        }
        if (is_contiguous()) {
            for (const auto& v : *data_) {
                fn(v);
            }
            return;
        }
        detail::Odometer od(shape);
        while (!od.done()) {
            fn((*data_)[_flat(od.idx())]);
            od.advance();
        }
    }

    template <typename T>
    template <typename Fn>
    void Ndarray<T>::_for_each_indexed(Fn&& fn) const {
        if (!data_) {
            return;
        }
        detail::Odometer od(shape);
        while (!od.done()) {
            const auto& idx = od.idx();
            fn(idx, (*data_)[_flat(idx)]);
            od.advance();
        }
    }

    template <typename T>
    void Ndarray<T>::_finalize() {
        strides = _c_strides(shape);
        if (!data_) {
            data_ = std::make_shared<std::vector<T>>(_numel(), T{});
        }
        if (type == dtype::void_) {
            type = dtype_of<T>;
        }
        order = matrix::Order::C;
    }

    // ---------------------------------------------------------------------
    // Reductions
    // ---------------------------------------------------------------------

    template <typename T>
    template <typename Acc, typename StepFn>
    auto Ndarray<T>::_reduce_axis(int axis, bool keepdims,
                                  std::optional<Acc> seed,
                                  StepFn&& step) const -> Ndarray<Acc> {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());

        std::vector<int> out_shape = shape;
        out_shape.erase(out_shape.begin() + axis);
        if (keepdims) {
            out_shape.insert(out_shape.begin() + axis, 1);
        }

        Ndarray<Acc> out(out_shape);
        if (seed.has_value()) {
            std::fill(out.data().begin(), out.data().end(), *seed);
        }
        std::vector<std::uint8_t> first(out.size(), seed.has_value() ? 0u : 1u);

        std::vector<std::size_t> out_idx;
        out_idx.reserve(nd - 1);
        detail::Odometer od(shape);
        while (!od.done()) {
            const auto& idx = od.idx();
            out_idx.clear();
            for (int d = 0; d < nd; ++d) {
                if (d != axis) {
                    out_idx.push_back(idx[d]);
                }
            }
            const std::size_t of =
                detail::flat_index(out_idx, out.strides, 0);
            const T value = (*data_)[_flat(idx)];
            if (first[of]) {
                out.data()[of] = static_cast<Acc>(value);
                first[of] = 0;
            } else {
                step(out.data()[of], value);
            }
            od.advance();
        }
        return out;
    }

    template <typename T>
    auto Ndarray<T>::sum() const
        -> std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T> {
        using Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        Acc total{};
        _for_each_logical([&](const T& v) { total += v; });
        return total;
    }

    template <typename T>
    template <typename Acc>
    auto Ndarray<T>::sum(int axis, bool keepdims) const -> Ndarray<Acc> {
        return _reduce_axis<Acc>(
            axis, keepdims, Acc(0), [](Acc& acc, const T& v) { acc += v; });
    }

    template <typename T>
    auto Ndarray<T>::prod() const
        -> std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T> {
        using Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        Acc total{1};
        _for_each_logical([&](const T& v) { total *= v; });
        return total;
    }

    template <typename T>
    template <typename Acc>
    auto Ndarray<T>::prod(int axis, bool keepdims) const -> Ndarray<Acc> {
        return _reduce_axis<Acc>(
            axis, keepdims, Acc(1), [](Acc& acc, const T& v) { acc *= v; });
    }

    template <typename T>
    T Ndarray<T>::min() const {
        if (_numel() == 0) {
            throw std::runtime_error("min() on empty array");
        }
        std::optional<T> best;
        _for_each_logical([&](const T& v) {
            if (!best.has_value() || v < *best) {
                best = v;
            }
        });
        return *best;
    }

    template <typename T>
    auto Ndarray<T>::min(int axis, bool keepdims) const -> Ndarray<T> {
        return _reduce_axis<T>(axis, keepdims, std::nullopt,
                               [](T& acc, const T& v) { acc = std::min(acc, v); });
    }

    template <typename T>
    T Ndarray<T>::max() const {
        if (_numel() == 0) {
            throw std::runtime_error("max() on empty array");
        }
        std::optional<T> best;
        _for_each_logical([&](const T& v) {
            if (!best.has_value() || v > *best) {
                best = v;
            }
        });
        return *best;
    }

    template <typename T>
    auto Ndarray<T>::max(int axis, bool keepdims) const -> Ndarray<T> {
        return _reduce_axis<T>(axis, keepdims, std::nullopt,
                               [](T& acc, const T& v) { acc = std::max(acc, v); });
    }

    template <typename T>
    auto Ndarray<T>::mean() const -> typename _mean_type<T>::type {
        using MeanT = typename _mean_type<T>::type;
        if (_numel() == 0) {
            throw std::runtime_error("mean() on empty array");
        }
        long double total = 0;
        _for_each_logical([&](const T& v) { total += static_cast<long double>(v); });
        return static_cast<MeanT>(total / static_cast<long double>(_numel()));
    }

    template <typename T>
    auto Ndarray<T>::mean(int axis, bool keepdims) const
        -> Ndarray<typename _mean_type<T>::type> {
        using MeanT = typename _mean_type<T>::type;
        axis = _normalize_axis(axis);
        const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
        auto s = _reduce_axis<MeanT>(axis, keepdims, MeanT(0),
                                     [](MeanT& acc, const T& v) { acc += v; });
        for (auto& v : s.data()) {
            v /= static_cast<MeanT>(axis_len);
        }
        return s;
    }

    template <typename T>
    template <typename MeanT>
    auto Ndarray<T>::_var_axis(int axis, bool keepdims) const
        -> Ndarray<MeanT> {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());

        std::vector<int> out_shape = shape;
        out_shape.erase(out_shape.begin() + axis);
        if (keepdims) {
            out_shape.insert(out_shape.begin() + axis, 1);
        }
        Ndarray<MeanT> out(out_shape);
        const std::size_t n_out = out.size();
        std::vector<long double> m(n_out, 0.0L), m2(n_out, 0.0L);
        std::vector<std::size_t> count(n_out, 0);

        std::vector<std::size_t> out_idx;
        out_idx.reserve(nd - 1);
        detail::Odometer od(shape);
        while (!od.done()) {
            const auto& idx = od.idx();
            out_idx.clear();
            for (int d = 0; d < nd; ++d) {
                if (d != axis) {
                    out_idx.push_back(idx[d]);
                }
            }
            const std::size_t of =
                detail::flat_index(out_idx, out.strides, 0);
            const long double v =
                static_cast<long double>((*data_)[_flat(idx)]);
            ++count[of];
            const long double delta = v - m[of];
            m[of] += delta / static_cast<long double>(count[of]);
            m2[of] += delta * (v - m[of]);
            od.advance();
        }
        for (std::size_t i = 0; i < n_out; ++i) {
            const long double denom =
                count[i] == 0 ? 1.0L : static_cast<long double>(count[i]);
            out.data()[i] = static_cast<MeanT>(m2[i] / denom);
        }
        return out;
    }

    template <typename T>
    auto Ndarray<T>::var() const -> typename _mean_type<T>::type {
        using MeanT = typename _mean_type<T>::type;
        if (_numel() == 0) {
            throw std::runtime_error("var() on empty array");
        }
        long double m = 0.0L, m2 = 0.0L;
        std::size_t count = 0;
        _for_each_logical([&](const T& v) {
            ++count;
            const long double x = static_cast<long double>(v);
            const long double delta = x - m;
            m += delta / static_cast<long double>(count);
            m2 += delta * (x - m);
        });
        return static_cast<MeanT>(m2 / static_cast<long double>(count));
    }

    template <typename T>
    auto Ndarray<T>::var(int axis, bool keepdims) const
        -> Ndarray<typename _mean_type<T>::type> {
        return _var_axis<typename _mean_type<T>::type>(axis, keepdims);
    }

    template <typename T>
    auto Ndarray<T>::std() const -> typename _mean_type<T>::type {
        return static_cast<typename _mean_type<T>::type>(std::sqrt(var()));
    }

    template <typename T>
    auto Ndarray<T>::std(int axis, bool keepdims) const
        -> Ndarray<typename _mean_type<T>::type> {
        auto v = _var_axis<typename _mean_type<T>::type>(axis, keepdims);
        for (auto& x : v.data()) {
            x = static_cast<typename _mean_type<T>::type>(std::sqrt(x));
        }
        return v;
    }

    template <typename T>
    bool Ndarray<T>::all() const {
        bool result = true;
        _for_each_logical([&](const T& v) { result = result && (v != T{}); });
        return result;
    }

    template <typename T>
    auto Ndarray<T>::all(int axis, bool keepdims) const -> Ndarray<bool> {
        return _reduce_axis<bool>(
            axis, keepdims, std::optional<bool>(true),
            [](bool& acc, const T& v) { acc = acc && (v != T{}); });
    }

    template <typename T>
    bool Ndarray<T>::any() const {
        bool result = false;
        _for_each_logical([&](const T& v) { result = result || (v != T{}); });
        return result;
    }

    template <typename T>
    auto Ndarray<T>::any(int axis, bool keepdims) const -> Ndarray<bool> {
        return _reduce_axis<bool>(
            axis, keepdims, std::optional<bool>(false),
            [](bool& acc, const T& v) { acc = acc || (v != T{}); });
    }

    template <typename T>
    template <typename Cmp>
    auto Ndarray<T>::_arg_reduce_axis(int axis, bool keepdims, Cmp&& cmp) const
        -> Ndarray<std::size_t> {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());

        std::vector<int> out_shape = shape;
        out_shape.erase(out_shape.begin() + axis);
        if (keepdims) {
            out_shape.insert(out_shape.begin() + axis, 1);
        }
        Ndarray<std::size_t> out(out_shape);
        std::vector<std::uint8_t> first(out.size(), 1u);
        std::vector<T> best_val(out.size(), T{});
        std::vector<std::size_t> best_pos(out.size(), 0);

        std::vector<std::size_t> out_idx;
        out_idx.reserve(nd - 1);
        detail::Odometer od(shape);
        while (!od.done()) {
            const auto& idx = od.idx();
            out_idx.clear();
            for (int d = 0; d < nd; ++d) {
                if (d != axis) {
                    out_idx.push_back(idx[d]);
                }
            }
            const std::size_t of =
                detail::flat_index(out_idx, out.strides, 0);
            const T value = (*data_)[_flat(idx)];
            if (first[of] || cmp(value, best_val[of])) {
                first[of] = 0;
                best_val[of] = value;
                best_pos[of] = idx[axis];
            }
            od.advance();
        }
        for (std::size_t i = 0; i < out.size(); ++i) {
            out.data()[i] = best_pos[i];
        }
        return out;
    }

    template <typename T>
    std::size_t Ndarray<T>::argmax() const {
        if (_numel() == 0) {
            throw std::runtime_error("argmax() on empty array");
        }
        std::size_t best = 0;
        std::size_t pos = 0;
        std::optional<T> best_val;
        detail::Odometer od(shape);
        while (!od.done()) {
            const T v = (*data_)[_flat(od.idx())];
            if (!best_val.has_value() || v > *best_val) {
                best_val = v;
                best = pos;
            }
            ++pos;
            od.advance();
        }
        return best;
    }

    template <typename T>
    auto Ndarray<T>::argmax(int axis, bool keepdims) const
        -> Ndarray<std::size_t> {
        return _arg_reduce_axis(axis, keepdims,
                                [](const T& v, const T& b) { return v > b; });
    }

    template <typename T>
    std::size_t Ndarray<T>::argmin() const {
        if (_numel() == 0) {
            throw std::runtime_error("argmin() on empty array");
        }
        std::size_t best = 0;
        std::size_t pos = 0;
        std::optional<T> best_val;
        detail::Odometer od(shape);
        while (!od.done()) {
            const T v = (*data_)[_flat(od.idx())];
            if (!best_val.has_value() || v < *best_val) {
                best_val = v;
                best = pos;
            }
            ++pos;
            od.advance();
        }
        return best;
    }

    template <typename T>
    auto Ndarray<T>::argmin(int axis, bool keepdims) const
        -> Ndarray<std::size_t> {
        return _arg_reduce_axis(axis, keepdims,
                                [](const T& v, const T& b) { return v < b; });
    }

    template <typename T>
    template <typename Acc, typename Fn>
    auto Ndarray<T>::_cum_axis(int axis, Fn&& fn) const -> Ndarray<Acc> {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());
        const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);

        Ndarray<Acc> out(shape);
        std::vector<int> reduced_shape = shape;
        reduced_shape.erase(reduced_shape.begin() + axis);
        const std::vector<std::size_t> red_strides =
            _c_strides(reduced_shape);
        const std::size_t n_slots = _numel() / axis_len;
        std::vector<Acc> acc(n_slots, Acc{});

        std::vector<std::size_t> slot;
        slot.reserve(nd - 1);
        std::vector<std::size_t> out_idx;
        out_idx.reserve(nd);
        detail::Odometer od(shape);
        while (!od.done()) {
            const auto& idx = od.idx();
            out_idx = idx;
            slot.clear();
            for (int d = 0; d < nd; ++d) {
                if (d != axis) {
                    slot.push_back(idx[d]);
                }
            }
            const std::size_t slot_of =
                detail::flat_index(slot, red_strides, 0);
            acc[slot_of] = fn(acc[slot_of], (*data_)[_flat(idx)]);
            out.data()[detail::flat_index(out_idx, out.strides, 0)] =
                acc[slot_of];
            od.advance();
        }
        return out;
    }

    template <typename T>
    auto Ndarray<T>::cumsum() const
        -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>> {
        using Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        Ndarray<Acc> out(std::vector<int>{static_cast<int>(_numel())});
        Acc running{};
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            running += v;
            out.data()[i++] = running;
        });
        return out;
    }

    template <typename T>
    auto Ndarray<T>::cumsum(int axis) const
        -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>> {
        using Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        return _cum_axis<Acc>(axis, [](Acc& acc, const T& v) { return acc + v; });
    }

    template <typename T>
    auto Ndarray<T>::cumprod() const
        -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>> {
        using Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        Ndarray<Acc> out(std::vector<int>{static_cast<int>(_numel())});
        Acc running{1};
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            running *= v;
            out.data()[i++] = running;
        });
        return out;
    }

    template <typename T>
    auto Ndarray<T>::cumprod(int axis) const
        -> Ndarray<std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>> {
        using Acc = std::conditional_t<std::is_same_v<T, bool>, std::int64_t, T>;
        return _cum_axis<Acc>(axis, [](Acc& acc, const T& v) { return acc * v; });
    }

    // ---------------------------------------------------------------------
    // Sorting / searching
    // ---------------------------------------------------------------------

    template <typename T>
    void Ndarray<T>::sort(int axis) {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());
        const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);

        std::vector<int> slice_shape = shape;
        slice_shape.erase(slice_shape.begin() + axis);

        std::vector<std::size_t> full(nd);
        detail::Odometer od(slice_shape);
        while (!od.done()) {
            const auto& s = od.idx();
            std::vector<T> work(axis_len);
            for (std::size_t p = 0; p < axis_len; ++p) {
                std::size_t f = 0;
                for (int d = 0; d < nd; ++d) {
                    full[d] = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                    f += full[d] * strides[d];
                }
                work[p] = (*data_)[offset + f];
            }
            std::stable_sort(work.begin(), work.end());
            for (std::size_t p = 0; p < axis_len; ++p) {
                std::size_t f = 0;
                for (int d = 0; d < nd; ++d) {
                    full[d] = (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                    f += full[d] * strides[d];
                }
                (*data_)[offset + f] = work[p];
            }
            od.advance();
        }
    }

    template <typename T>
    auto Ndarray<T>::sorted(int axis) const -> Ndarray<T> {
        Ndarray<T> out = *this;
        out.sort(axis);
        return out;
    }

    template <typename T>
    auto Ndarray<T>::argsort(int axis) const -> Ndarray<std::size_t> {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());
        const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);

        Ndarray<std::size_t> out(shape);
        std::vector<int> slice_shape = shape;
        slice_shape.erase(slice_shape.begin() + axis);

        std::vector<std::pair<std::size_t, T>> work;
        work.reserve(axis_len);
        detail::Odometer od(slice_shape);
        while (!od.done()) {
            const auto& s = od.idx();
            work.clear();
            for (std::size_t p = 0; p < axis_len; ++p) {
                std::size_t f = 0;
                for (int d = 0; d < nd; ++d) {
                    const std::size_t coord =
                        (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                    f += coord * strides[d];
                }
                work.emplace_back(p, (*data_)[offset + f]);
            }
            std::stable_sort(work.begin(), work.end(),
                             [](const auto& a, const auto& b) {
                                 return a.second < b.second;
                             });
            for (std::size_t p = 0; p < axis_len; ++p) {
                std::size_t f = 0;
                for (int d = 0; d < nd; ++d) {
                    const std::size_t coord =
                        (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                    f += coord * out.strides[d];
                }
                out.data()[f] = work[p].first;
            }
            od.advance();
        }
        return out;
    }

    template <typename T>
    auto Ndarray<T>::argpartition(std::size_t kth, int axis) const
        -> Ndarray<std::size_t> {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());
        const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
        if (kth >= axis_len) {
            throw std::out_of_range("kth out of bounds");
        }

        Ndarray<std::size_t> out(shape);
        std::vector<int> slice_shape = shape;
        slice_shape.erase(slice_shape.begin() + axis);

        std::vector<std::pair<std::size_t, T>> work;
        work.reserve(axis_len);
        detail::Odometer od(slice_shape);
        while (!od.done()) {
            const auto& s = od.idx();
            work.clear();
            for (std::size_t p = 0; p < axis_len; ++p) {
                std::size_t f = 0;
                for (int d = 0; d < nd; ++d) {
                    const std::size_t coord =
                        (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                    f += coord * strides[d];
                }
                work.emplace_back(p, (*data_)[offset + f]);
            }
            std::nth_element(work.begin(), work.begin() + kth, work.end(),
                             [](const auto& a, const auto& b) {
                                 return a.second < b.second;
                             });
            for (std::size_t p = 0; p < axis_len; ++p) {
                std::size_t f = 0;
                for (int d = 0; d < nd; ++d) {
                    const std::size_t coord =
                        (d < axis) ? s[d] : (d == axis ? p : s[d - 1]);
                    f += coord * out.strides[d];
                }
                out.data()[f] = work[p].first;
            }
            od.advance();
        }
        return out;
    }

    template <typename T>
    std::size_t Ndarray<T>::searchsorted(const T& value, bool side_right) const {
        if (shape.size() != 1) {
            throw std::invalid_argument("searchsorted requires a 1D array");
        }
        const auto first = begin();
        const auto last = end();
        const auto it = side_right
                            ? std::upper_bound(first, last, value)
                            : std::lower_bound(first, last, value);
        return static_cast<std::size_t>(std::distance(first, it));
    }

    template <typename T>
    auto Ndarray<T>::searchsorted(const Ndarray<int>& values) const
        -> Ndarray<std::size_t> {
        if (shape.size() != 1) {
            throw std::invalid_argument("searchsorted requires a 1D array");
        }
        Ndarray<std::size_t> out(
            std::vector<int>{static_cast<int>(values.size())});
        for (std::size_t i = 0; i < values.size(); ++i) {
            out.data()[i] = searchsorted(values.data()[values._flat_logical(i)]);
        }
        return out;
    }

    // ---------------------------------------------------------------------
    // Shape manipulation
    // ---------------------------------------------------------------------

    template <typename T>
    auto Ndarray<T>::reshape(const std::vector<int>& new_shape) const
        -> Ndarray {
        std::vector<int> resolved = new_shape;
        int neg_count = 0;
        for (int d : resolved) {
            if (d == -1) {
                ++neg_count;
            }
        }
        if (neg_count > 1) {
            throw std::invalid_argument("at most one dimension may be -1");
        }
        if (neg_count == 1) {
            std::size_t known = 1;
            int neg_at = 0;
            for (std::size_t i = 0; i < resolved.size(); ++i) {
                if (resolved[i] == -1) {
                    neg_at = static_cast<int>(i);
                } else {
                    known *= static_cast<std::size_t>(resolved[i]);
                }
            }
            if (known == 0 || _numel() % known != 0) {
                throw std::invalid_argument("cannot infer -1 dimension");
            }
            resolved[neg_at] = static_cast<int>(_numel() / known);
        }
        std::size_t total = 1;
        for (int d : resolved) {
            total *= static_cast<std::size_t>(d);
        }
        if (total != _numel()) {
            throw std::invalid_argument(
                "cannot reshape array of size " + std::to_string(_numel()) +
                " into shape with total size " + std::to_string(total));
        }
        if (is_contiguous()) {
            // View sharing storage
            return Ndarray(data_, resolved, _c_strides(resolved), type, order,
                           offset);
        }
        // Copy path
        Ndarray out(resolved, type);
        std::copy(begin(), end(), out.begin());
        return out;
    }

    template <typename T>
    auto Ndarray<T>::transpose() const -> Ndarray {
        if (shape.empty()) {
            return *this;
        }
        std::vector<int> p(shape.size());
        std::vector<std::size_t> s(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i) {
            p[i] = shape[shape.size() - 1 - i];
            s[i] = strides[strides.size() - 1 - i];
        }
        matrix::Order o =
            (order == matrix::Order::C) ? matrix::Order::F : matrix::Order::C;
        return Ndarray(data_, std::move(p), std::move(s), type, o, offset);
    }

    template <typename T>
    auto Ndarray<T>::transpose(const std::vector<int>& perm) const -> Ndarray {
        if (perm.size() != shape.size()) {
            throw std::invalid_argument(
                "permutation length must equal ndim");
        }
        std::vector<int> p(perm.size());
        std::vector<std::size_t> s(perm.size());
        std::vector<std::uint8_t> seen(perm.size(), 0);
        for (std::size_t i = 0; i < perm.size(); ++i) {
            int a = perm[i];
            if (a < 0) {
                a += static_cast<int>(perm.size());
            }
            if (a < 0 || a >= static_cast<int>(perm.size()) || seen[a]) {
                throw std::invalid_argument("invalid permutation");
            }
            seen[a] = 1;
            p[i] = shape[a];
            s[i] = strides[a];
        }
        return Ndarray(data_, std::move(p), std::move(s), type, order, offset);
    }

    template <typename T>
    auto Ndarray<T>::swapaxes(int axis1, int axis2) const -> Ndarray {
        axis1 = _normalize_axis(axis1);
        axis2 = _normalize_axis(axis2);
        std::vector<int> p = shape;
        std::vector<std::size_t> s = strides;
        std::swap(p[axis1], p[axis2]);
        std::swap(s[axis1], s[axis2]);
        return Ndarray(data_, std::move(p), std::move(s), type, order, offset);
    }

    template <typename T>
    auto Ndarray<T>::squeeze() const -> Ndarray {
        std::vector<int> p;
        std::vector<std::size_t> s;
        p.reserve(shape.size());
        s.reserve(shape.size());
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != 1) {
                p.push_back(shape[i]);
                s.push_back(strides[i]);
            }
        }
        if (p == shape) {
            return *this;
        }
        return Ndarray(data_, std::move(p), std::move(s), type, order, offset);
    }

    template <typename T>
    auto Ndarray<T>::squeeze(int axis) const -> Ndarray {
        axis = _normalize_axis(axis);
        if (shape[axis] != 1) {
            throw std::invalid_argument(
                "cannot squeeze a dimension that is not of size 1");
        }
        std::vector<int> p = shape;
        std::vector<std::size_t> s = strides;
        p.erase(p.begin() + axis);
        s.erase(s.begin() + axis);
        return Ndarray(data_, std::move(p), std::move(s), type, order, offset);
    }

    template <typename T>
    auto Ndarray<T>::ravel() const -> Ndarray {
        if (is_contiguous()) {
            return Ndarray(data_, {static_cast<int>(_numel())},
                           {std::size_t{1}}, type, order, offset);
        }
        return flatten();
    }

    template <typename T>
    auto Ndarray<T>::flatten() const -> Ndarray {
        Ndarray out({static_cast<int>(_numel())}, type);
        std::copy(begin(), end(), out.begin());
        return out;
    }

    template <typename T>
    void Ndarray<T>::resize(const std::vector<int>& new_shape) {
        std::size_t total = 1;
        for (int d : new_shape) {
            total *= static_cast<std::size_t>(d);
        }
        std::vector<T> flat;
        flat.reserve(total);
        _for_each_logical([&](const T& v) {
            if (flat.size() < total) {
                flat.push_back(v);
            }
        });
        flat.resize(total, T{});
        shape = new_shape;
        strides = _c_strides(new_shape);
        offset = 0;
        data_ = std::make_shared<std::vector<T>>(std::move(flat));
        type = type;
    }

    // ---------------------------------------------------------------------
    // Manipulation
    // ---------------------------------------------------------------------

    template <typename T>
    void Ndarray<T>::fill(const T& value) {
        if (!data_) {
            data_ = std::make_shared<std::vector<T>>(_numel(), value);
            return;
        }
        if (is_contiguous()) {
            std::fill(data_->begin(), data_->end(), value);
            return;
        }
        _for_each_indexed([&](const std::vector<std::size_t>& idx, const T&) {
            (*data_)[_flat(idx)] = value;
        });
    }

    template <typename T>
    auto Ndarray<T>::copy() const -> Ndarray {
        Ndarray out(shape, type);
        std::copy(begin(), end(), out.begin());
        return out;
    }

    template <typename T>
    auto Ndarray<T>::view() const -> Ndarray {
        return Ndarray(data_, shape, strides, type, order, offset);
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::astype() const -> Ndarray<U> {
        Ndarray<U> out(shape);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) { out.data()[i++] = static_cast<U>(v); });
        return out;
    }

    template <typename T>
    auto Ndarray<T>::take(const std::vector<std::size_t>& indices,
                          int axis) const -> Ndarray {
        const int nd = static_cast<int>(shape.size());
        axis = _normalize_axis(axis);
        std::vector<int> out_shape = shape;
        out_shape[axis] = static_cast<int>(indices.size());
        Ndarray out(out_shape, type);

        const std::size_t axis_len = static_cast<std::size_t>(shape[axis]);
        for (std::size_t k = 0; k < indices.size(); ++k) {
            if (indices[k] >= axis_len) {
                throw std::out_of_range("take index out of bounds");
            }
        }

        std::vector<int> slice_shape = shape;
        slice_shape.erase(slice_shape.begin() + axis);

        detail::Odometer od(slice_shape);
        while (!od.done()) {
            const auto& s = od.idx();
            for (std::size_t k = 0; k < indices.size(); ++k) {
                std::size_t in_f = 0, out_f = 0;
                for (int d = 0; d < nd; ++d) {
                    const std::size_t coord =
                        (d < axis) ? s[d] : (d == axis ? indices[k] : s[d - 1]);
                    in_f += coord * strides[d];
                    const std::size_t out_coord =
                        (d < axis) ? s[d] : (d == axis ? k : s[d - 1]);
                    out_f += out_coord * out.strides[d];
                }
                out.data()[out_f] = (*data_)[offset + in_f];
            }
            od.advance();
        }
        return out;
    }

    template <typename T>
    void Ndarray<T>::put(const std::vector<std::size_t>& indices,
                         const std::vector<T>& values, char mode) {
        const std::size_t n = _numel();
        for (std::size_t k = 0; k < indices.size(); ++k) {
            std::size_t p = indices[k];
            if (mode == 'w') {
                p %= n;
            } else if (mode == 'c') {
                p = std::min(p, n - 1);
            } else if (p >= n) {
                throw std::out_of_range("put index out of bounds");
            }
            const T& v = values.empty() ? T{} : values[k % values.size()];
            // logical flat index -> multi-index -> flat storage offset
            std::vector<std::size_t> idx = _shape_u();
            std::size_t rem = p;
            for (std::size_t d = shape.size(); d-- > 0;) {
                idx[d] = rem % static_cast<std::size_t>(shape[d]);
                rem /= static_cast<std::size_t>(shape[d]);
            }
            (*data_)[_flat(idx)] = v;
        }
    }

    template <typename T>
    auto Ndarray<T>::repeat(std::size_t repeats) const -> Ndarray {
        Ndarray out({static_cast<int>(_numel() * repeats)}, type);
        std::size_t o = 0;
        _for_each_logical([&](const T& v) {
            for (std::size_t r = 0; r < repeats; ++r) {
                out.data()[o++] = v;
            }
        });
        return out;
    }

    template <typename T>
    auto Ndarray<T>::repeat(std::size_t repeats, int axis) const -> Ndarray {
        axis = _normalize_axis(axis);
        const int nd = static_cast<int>(shape.size());
        std::vector<int> out_shape = shape;
        out_shape[axis] = static_cast<int>(
            static_cast<std::size_t>(shape[axis]) * repeats);
        Ndarray out(out_shape, type);

        detail::Odometer od(shape);
        while (!od.done()) {
            const auto& idx = od.idx();
            for (std::size_t r = 0; r < repeats; ++r) {
                std::size_t in_f = _flat(idx);
                std::size_t out_f = 0;
                for (int d = 0; d < nd; ++d) {
                    const std::size_t coord =
                        (d == axis) ? idx[d] * repeats + r : idx[d];
                    out_f += coord * out.strides[d];
                }
                out.data()[out_f] = (*data_)[in_f];
            }
            od.advance();
        }
        return out;
    }

    template <typename T>
    auto Ndarray<T>::clip(const T& min_value, const T& max_value) const
        -> Ndarray {
        Ndarray out(shape, type);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            out.data()[i++] = std::clamp(v, min_value, max_value);
        });
        return out;
    }

    template <typename T>
    auto Ndarray<T>::round(int decimals) const -> Ndarray {
        Ndarray out(shape, type);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            if constexpr (std::is_floating_point_v<T>) {
                const T factor =
                    static_cast<T>(std::pow(10.0, static_cast<double>(decimals)));
                out.data()[i++] = std::round(v * factor) / factor;
            } else {
                out.data()[i++] = v;
            }
        });
        return out;
    }

    template <typename T>
    auto Ndarray<T>::diagonal(int offset) const -> Ndarray {
        if (shape.size() < 2) {
            throw np::AxisError("diagonal requires an array with ndim >= 2");
        }
        const std::size_t n0 = static_cast<std::size_t>(shape[0]);
        const std::size_t n1 = static_cast<std::size_t>(shape[1]);

        std::size_t len = 0;
        if (offset >= 0) {
            const std::size_t o = static_cast<std::size_t>(offset);
            len = (n1 > o) ? std::min(n0, n1 - o) : 0;
        } else {
            const std::size_t o = static_cast<std::size_t>(-offset);
            len = (n0 > o) ? std::min(n1, n0 - o) : 0;
        }

        std::vector<int> out_shape;
        out_shape.push_back(static_cast<int>(len));
        out_shape.insert(out_shape.end(), shape.begin() + 2, shape.end());
        Ndarray out(out_shape, type);

        detail::Odometer od(out_shape);
        while (!od.done()) {
            const auto& oi = od.idx();
            std::vector<std::size_t> in_idx(shape.size());
            in_idx[0] = oi[0];
            in_idx[1] = oi[0] + static_cast<std::size_t>(offset);
            for (std::size_t d = 2; d < shape.size(); ++d) {
                in_idx[d] = oi[d - 1];
            }
            out.data()[detail::flat_index(oi, out.strides, 0)] =
                (*data_)[_flat(in_idx)];
            od.advance();
        }
        return out;
    }

    template <typename T>
    T Ndarray<T>::trace(int offset) const {
        if (shape.size() < 2) {
            throw np::AxisError("trace requires an array with ndim >= 2");
        }
        auto diag = diagonal(offset);
        T total{};
        for (const auto& v : diag) {
            total += v;
        }
        return total;
    }

    template <typename T>
    auto Ndarray<T>::nonzero() const -> std::vector<Ndarray<std::size_t>> {
        std::vector<Ndarray<std::size_t>> result(shape.size());
        std::vector<std::vector<std::size_t>> per_dim(shape.size());
        _for_each_indexed([&](const std::vector<std::size_t>& idx, const T& v) {
            if (v != T{}) {
                for (std::size_t d = 0; d < idx.size(); ++d) {
                    per_dim[d].push_back(idx[d]);
                }
            }
        });
        for (std::size_t d = 0; d < result.size(); ++d) {
            const int n_coords = static_cast<int>(per_dim[d].size());
            result[d] = Ndarray<std::size_t>::from_data(
                std::vector<int>{n_coords}, std::move(per_dim[d]));
        }
        return result;
    }

    template <typename T>
    auto Ndarray<T>::conj() const -> Ndarray {
        Ndarray out(shape, type);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            if constexpr (detail::is_complex_v<T>) {
                out.data()[i++] = std::conj(v);
            } else {
                out.data()[i++] = v;
            }
        });
        return out;
    }

    template <typename T>
    void Ndarray<T>::byteswap() {
        if (!data_) {
            return;
        }
        if (is_contiguous()) {
            for (auto& v : *data_) {
                char* p = reinterpret_cast<char*>(&v);
                std::reverse(p, p + sizeof(T));
            }
            return;
        }
        _for_each_indexed([&](const std::vector<std::size_t>& idx, const T&) {
            T& v = (*data_)[_flat(idx)];
            char* p = reinterpret_cast<char*>(&v);
            std::reverse(p, p + sizeof(T));
        });
    }

    // ---------------------------------------------------------------------
    // Conversions / IO
    // ---------------------------------------------------------------------

    template <typename T>
    auto Ndarray<T>::tolist() const -> std::vector<T> {
        return std::vector<T>(begin(), end());
    }

    template <typename T>
    auto Ndarray<T>::tobytes() const -> std::vector<std::uint8_t> {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(_numel() * sizeof(T));
        _for_each_logical([&](const T& v) {
            const std::uint8_t* p =
                reinterpret_cast<const std::uint8_t*>(&v);
            bytes.insert(bytes.end(), p, p + sizeof(T));
        });
        return bytes;
    }

    template <typename T>
    void Ndarray<T>::tofile(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            throw std::runtime_error("cannot open file: " + filename);
        }
        auto bytes = tobytes();
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }

    template <typename T>
    void Ndarray<T>::tofile(std::ostream& os) const {
        auto bytes = tobytes();
        os.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    }

    template <typename T>
    void Ndarray<T>::print(std::ostream& os) const {
        _print_to(os);
        os << '\n';
    }

    template <typename T>
    void Ndarray<T>::_print_recursive(std::size_t dim, std::size_t flat_offset,
                                      std::ostream& os) const {
        if (shape.empty()) {
            os << (*data_)[offset];
            return;
        }
        if (dim == shape.size() - 1) {
            os << "[";
            for (std::size_t i = 0; i < static_cast<std::size_t>(shape[dim]);
                 ++i) {
                if (i != 0) {
                    os << ", ";
                }
                os << (*data_)[flat_offset + i * strides[dim]];
            }
            os << "]";
            return;
        }
        os << "[";
        for (std::size_t i = 0; i < static_cast<std::size_t>(shape[dim]);
             ++i) {
            if (i != 0) {
                os << ",\n ";
            }
            _print_recursive(dim + 1,
                             flat_offset + i * strides[dim], os);
        }
        os << "]";
    }

    template <typename T>
    void Ndarray<T>::_print_to(std::ostream& os) const {
        if (!data_) {
            os << "array([])";
            return;
        }
        os << "array(";
        _print_recursive(0, offset, os);
        os << ", dtype=" << dtype_name(type) << ")";
    }

    // ---------------------------------------------------------------------
    // Element-wise arithmetic
    // ---------------------------------------------------------------------

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator+(const Ndarray<U>& rhs) const
        -> Ndarray<std::common_type_t<T, U>> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a + b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator-(const Ndarray<U>& rhs) const
        -> Ndarray<std::common_type_t<T, U>> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a - b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator*(const Ndarray<U>& rhs) const
        -> Ndarray<std::common_type_t<T, U>> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a * b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator/(const Ndarray<U>& rhs) const
        -> Ndarray<std::common_type_t<T, U>> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a / b; });
    }

    template <typename T>
    template <typename U, typename Fn>
    auto Ndarray<T>::_scalar_op(const U& scalar, Fn&& fn) const
        -> Ndarray<std::common_type_t<T, U>> {
        using R = std::common_type_t<T, U>;
        Ndarray<R> out(shape);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            out.data()[i++] = fn(v, scalar);
        });
        return out;
    }

    template <typename T>
    template <typename U, typename Fn>
    auto Ndarray<T>::_scalar_left_op(const U& scalar, Fn&& fn) const
        -> Ndarray<std::common_type_t<U, T>> {
        using R = std::common_type_t<U, T>;
        Ndarray<R> out(shape);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            out.data()[i++] = fn(scalar, v);
        });
        return out;
    }

    template <typename T>
    template <typename U, typename Fn>
    auto Ndarray<T>::_cmp_scalar(const U& scalar, Fn&& fn) const
        -> Ndarray<bool> {
        Ndarray<bool> out(shape, dtype::bool_);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) {
            out.data()[i++] = fn(v, scalar);
        });
        return out;
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator+(const U& scalar) const
        -> Ndarray<std::common_type_t<T, U>> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _scalar_op(scalar, [](const T& a, const U& b) { return a + b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator-(const U& scalar) const
        -> Ndarray<std::common_type_t<T, U>> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _scalar_op(scalar, [](const T& a, const U& b) { return a - b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator*(const U& scalar) const
        -> Ndarray<std::common_type_t<T, U>> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _scalar_op(scalar, [](const T& a, const U& b) { return a * b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator/(const U& scalar) const
        -> Ndarray<std::common_type_t<T, U>> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _scalar_op(scalar, [](const T& a, const U& b) { return a / b; });
    }

    template <typename T>
    auto Ndarray<T>::operator-() const -> Ndarray {
        Ndarray out(shape, type);
        std::size_t i = 0;
        _for_each_logical([&](const T& v) { out.data()[i++] = -v; });
        return out;
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator==(const Ndarray<U>& rhs) const -> Ndarray<bool> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a == b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator!=(const Ndarray<U>& rhs) const -> Ndarray<bool> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a != b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator<(const Ndarray<U>& rhs) const -> Ndarray<bool> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a < b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator<=(const Ndarray<U>& rhs) const -> Ndarray<bool> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a <= b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator>(const Ndarray<U>& rhs) const -> Ndarray<bool> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a > b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator>=(const Ndarray<U>& rhs) const -> Ndarray<bool> {
        return detail::elementwise(
            *this, rhs, [](const T& a, const U& b) { return a >= b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator==(const U& scalar) const -> Ndarray<bool> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _cmp_scalar(scalar, [](const T& a, const U& b) { return a == b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator!=(const U& scalar) const -> Ndarray<bool> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _cmp_scalar(scalar, [](const T& a, const U& b) { return a != b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator<(const U& scalar) const -> Ndarray<bool> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _cmp_scalar(scalar, [](const T& a, const U& b) { return a < b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator<=(const U& scalar) const -> Ndarray<bool> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _cmp_scalar(scalar, [](const T& a, const U& b) { return a <= b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator>(const U& scalar) const -> Ndarray<bool> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _cmp_scalar(scalar, [](const T& a, const U& b) { return a > b; });
    }

    template <typename T>
    template <typename U>
    auto Ndarray<T>::operator>=(const U& scalar) const -> Ndarray<bool> {
        static_assert(_is_valid_scalar<U>,
                      "scalar operand must be arithmetic or complex");
        return _cmp_scalar(scalar, [](const T& a, const U& b) { return a >= b; });
    }

    template <typename T>
    bool Ndarray<T>::all_equal(const Ndarray& other) const noexcept {
        if (shape != other.shape || !data_ || !other.data_) {
            return false;
        }
        try {
            detail::Odometer od(shape);
            while (!od.done()) {
                const auto& idx = od.idx();
                if (!((*data_)[_flat(idx)] ==
                      (*other.data_)[other._flat(idx)])) {
                    return false;
                }
                od.advance();
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    template <typename T>
    bool Ndarray<T>::all_equal(const T& value) const noexcept {
        try {
            detail::Odometer od(shape);
            while (!od.done()) {
                const auto& idx = od.idx();
                if (!((*data_)[_flat(idx)] == value)) {
                    return false;
                }
                od.advance();
            }
        } catch (...) {
            return false;
        }
        return true;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator+=(const Ndarray& rhs) {
        *this = *this + rhs;
        return *this;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator-=(const Ndarray& rhs) {
        *this = *this - rhs;
        return *this;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator*=(const Ndarray& rhs) {
        *this = *this * rhs;
        return *this;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator/=(const Ndarray& rhs) {
        *this = *this / rhs;
        return *this;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator+=(const T& scalar) {
        *this = *this + scalar;
        return *this;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator-=(const T& scalar) {
        *this = *this - scalar;
        return *this;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator*=(const T& scalar) {
        *this = *this * scalar;
        return *this;
    }

    template <typename T>
    Ndarray<T>& Ndarray<T>::operator/=(const T& scalar) {
        *this = *this / scalar;
        return *this;
    }

} // namespace np

#endif // NP_NDARRAY_HPP
