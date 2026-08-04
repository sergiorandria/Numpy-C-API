/**
 * @file test_compile_time.cpp
 * @brief Negative compile-time guarantees of the fixed-shape path.
 *
 *        Every invalid use below (shape mismatch, unbroadcastable operands,
 *        element-count-preserving reshape violations, mismatched joins,
 *        out-of-range axes, wrong indexing arity) is expressed as a
 *        requires-expression over the public API. The static_asserts prove
 *        the library REJECTS these programs at compile time instead of
 *        silently truncating / misaligning data (NumPy parity: those are
 *        runtime errors there, but the fixed path turns them into
 *        compile-time errors).
 */
#include "np/np.hpp"

namespace {

using np::ndarray;

template <typename L, typename R>
concept addable = requires(const L& l, const R& r) { l + r; };

template <typename A, typename B>
concept concatable = requires(const A& a, const B& b) {
    np::concatenate(a, b);
};

template <typename A, typename B>
concept stackable = requires(const A& a, const B& b) {
    np::stack<0>(a, b);
};

template <typename A, int... E>
concept reshapable = requires(const A& a) {
    a.template reshape<E...>();
};

template <typename A, int Axis>
concept squeezable = requires(const A& a) {
    a.template squeeze<Axis>();
};

template <typename A, int Axis>
concept expandable = requires(const A& a) {
    a.template expand_dims<Axis>();
};

template <typename A, int Axis>
concept summable = requires(const A& a) {
    a.template sum<Axis>();
};

template <typename A>
concept three_index = requires(const A& a) { a(0, 1, 2); };

template <typename A, typename B>
concept dottable = requires(const A& a, const B& b) {
    np::linalg::dot(a, b);
};

template <typename A, typename B>
concept matmulable = requires(const A& a, const B& b) {
    np::linalg::matmul(a, b);
};

// --- the detector itself is sound: valid uses are accepted -----------------
static_assert(addable<ndarray<int, 2, 3>, ndarray<int, 2, 3>>);
static_assert(addable<ndarray<int, 2, 3>, ndarray<int, 3>>);
static_assert(addable<ndarray<int, 2, 3>, ndarray<int, 2, 1>>);
static_assert(addable<ndarray<int, 2, 3>, ndarray<int, 1, 3>>);
static_assert(addable<ndarray<int, 2, 3>, int>);
static_assert(addable<int, ndarray<int, 2, 3>>);
static_assert(reshapable<ndarray<int, 2, 3>, 3, 2>);
static_assert(reshapable<ndarray<int, 2, 3>, 1, 6>);
static_assert(reshapable<ndarray<int, 2, 3>, 2, 3>);
static_assert(concatable<ndarray<int, 3>, ndarray<int, 2>>);
static_assert(concatable<ndarray<int, 2, 3>, ndarray<int, 4, 3>>);
static_assert(stackable<ndarray<int, 2, 2>, ndarray<int, 2, 2>>);
static_assert(squeezable<ndarray<int, 1, 3, 1>, 0>);
static_assert(expandable<ndarray<int, 2, 3>, 0>);
static_assert(expandable<ndarray<int, 2, 3>, 2>);
static_assert(summable<ndarray<int, 2, 3>, 1>);
static_assert(three_index<ndarray<int, 2, 3, 4>>);
static_assert(dottable<ndarray<int, 3>, ndarray<int, 3>>);
static_assert(dottable<ndarray<int, 2, 3>, ndarray<int, 3>>);
static_assert(dottable<ndarray<int, 3>, ndarray<int, 3, 2>>);
static_assert(dottable<ndarray<int, 2, 3>, ndarray<int, 3, 2>>);
static_assert(matmulable<ndarray<int, 2, 3>, ndarray<int, 3, 2>>);

// --- shape mismatches are rejected at compile time -------------------------
static_assert(!addable<ndarray<int, 2, 3>, ndarray<int, 2, 2>>);
static_assert(!addable<ndarray<int, 2, 3>, ndarray<int, 4>>);
static_assert(!addable<ndarray<int, 2, 3>, ndarray<int, 3, 2>>);
static_assert(!addable<ndarray<int, 2, 3>, ndarray<int, 5, 4, 3>>);

// --- reshape must preserve the element count -------------------------------
static_assert(!reshapable<ndarray<int, 2, 3>, 2, 2>);
static_assert(!reshapable<ndarray<int, 2, 3>, 7>);
static_assert(!reshapable<ndarray<int, 2, 3>, 0, 5>);
static_assert(!reshapable<ndarray<int, 2, 3>, 2, 2, 1>);

// --- joins must be shape-consistent ----------------------------------------
static_assert(!concatable<ndarray<int, 2, 3>, ndarray<int, 4>>);
static_assert(!concatable<ndarray<int, 3>, ndarray<int, 2, 2>>);
static_assert(!stackable<ndarray<int, 2, 3>, ndarray<int, 3>>);
static_assert(!stackable<ndarray<int, 2, 2>, ndarray<int, 3, 2>>);

// --- axes must be in range --------------------------------------------------
static_assert(!squeezable<ndarray<int, 2, 3>, 5>);
static_assert(!expandable<ndarray<int, 2, 3>, 7>);
static_assert(!summable<ndarray<int, 2, 3>, 5>);
static_assert(!summable<ndarray<int, 2, 3>, -2>);

// --- indexing arity must match the rank ------------------------------------
static_assert(!three_index<ndarray<int, 2, 3>>);
static_assert(!three_index<ndarray<int, 4>>);

// --- linalg contraction dimension is checked at compile time ---------------
static_assert(!dottable<ndarray<int, 2, 3>, ndarray<int, 2, 3>>);
static_assert(!dottable<ndarray<int, 3, 2>, ndarray<int, 3>>);
static_assert(!matmulable<ndarray<int, 2, 3>, ndarray<int, 2, 4>>);
static_assert(!matmulable<ndarray<int, 2, 2>, ndarray<int, 3, 2>>);

} // namespace

int main() { return 0; }
