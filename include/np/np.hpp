/**
 * @file np.hpp
 * @brief Umbrella header for the entire NumPy-like C++ library.
 *
 * Include this single header to get the whole API:
 *   #include <np/np.hpp>
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_NP_HPP
#define NP_NP_HPP

#include "api_macros.hpp"
#include "simd.hpp"
#include "bitwise.hpp"
#include "char.hpp"
#include "constants.hpp"
#include "creation.hpp"
#include "creation_fixed.hpp"
#include "datetime.hpp"
#include "dtype.hpp"
#include "emath.hpp"
#include "err.hpp"
#include "exceptions.hpp"
#include "fft.hpp"
#include "functional.hpp"
#include "linalg.hpp"
#include "linalg_fixed.hpp"
#include "logic.hpp"
#include "concatenate.hpp"
#include "manipulation.hpp"
#include "masked_array.hpp"
#include "math.hpp"
#include "matrix.hpp"
#include "ndarray.hpp"
#include "ndarray_fixed.hpp"
#include "sorting.hpp"
#include "statistics.hpp"
#include "testing.hpp"
#include "window.hpp"
#include "io.hpp"
#include "polynomial.hpp"
#include "indexing.hpp"
#include "other.hpp"
#include "pqc.hpp"
#include "threadpool.hpp"
#include "bigint.hpp"
#include "homology.hpp"
#include "homotopy.hpp"
#include "modular.hpp"
#include "manifold.hpp"
#include "variety.hpp"
#include "differential.hpp"
#include "cohomology.hpp"
#include "bundle.hpp"
#include "persistent.hpp"
#include "spectral.hpp"
#include "lattice.hpp"
#include "padic.hpp"
#include "neuromorphic.hpp"
#include "memory.hpp"
#include "tensor_core.hpp"
#include "random.hpp"

// Suppress -Wbraced-scalar-init for NDProxy braced-init (e.g.
// {{{1},{2},{3}},{{1},{2},{3}}} shape 2×3×1)
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wbraced-scalar-init"
#endif

#endif // NP_NP_HPP
