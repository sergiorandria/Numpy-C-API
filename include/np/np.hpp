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
#include "variety.hpp"
#include "differential.hpp"
// Note: random.hpp and concatenate.hpp are not included by default.
// Include them explicitly if needed to avoid template conflicts.

#endif // NP_NP_HPP
