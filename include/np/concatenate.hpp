/**
 * @file concatenate.hpp
 * @brief Array concatenation and stacking routines.
 *
 * Provides NumPy-compatible joining operations:
 *   concatenate, stack, vstack, hstack, dstack, column_stack, row_stack
 *
 * Reference: numpy-reference/reference/routines.array-manipulation.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_CONCATENATE_HPP
#define NP_CONCATENATE_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>

#include "ndarray.hpp"

namespace np {

    // =================================================================
    // Concatenate
    // Reference: numpy-reference/reference/generated/numpy.concatenate.html
    // =================================================================

    /**
     * @brief Join a sequence of arrays along an existing axis.
     *
     * All arrays must have the same shape except in the concatenation axis.
     *
     * @param arrays Sequence of arrays to concatenate.
     * @param axis Axis along which to concatenate (default: 0).
     * @throws std::invalid_argument if shapes are incompatible.
     */
    template <typename T>
    auto concatenate(const std::vector<Ndarray<T>>& arrays, int axis = 0)
        -> Ndarray<T> {
        if (arrays.empty()) {
            throw std::invalid_argument("concatenate: need at least one array");
        }
        
        const auto& first = arrays[0];
        const int ndim = static_cast<int>(first.ndim());
        
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            throw std::invalid_argument("concatenate: axis out of bounds");
        }
        
        // Check shape compatibility
        for (std::size_t i = 1; i < arrays.size(); ++i) {
            if (arrays[i].ndim() != first.ndim()) {
                throw std::invalid_argument(
                    "concatenate: all arrays must have same ndim");
            }
            for (int d = 0; d < ndim; ++d) {
                if (d != axis && arrays[i].shape[d] != first.shape[d]) {
                    throw std::invalid_argument(
                        "concatenate: shapes must match on non-concat axis");
                }
            }
        }
        
        // Compute output shape
        std::vector<int> out_shape = first.shape;
        for (std::size_t i = 1; i < arrays.size(); ++i) {
            out_shape[axis] += arrays[i].shape[axis];
        }
        
        // Allocate output
        Ndarray<T> result(out_shape, first.type);
        
        // Copy data
        std::vector<std::size_t> idx(ndim, 0);
        std::size_t offset = 0;
        
        for (const auto& arr : arrays) {
            const std::size_t axis_size = static_cast<std::size_t>(arr.shape[axis]);
            
            // Iterate over all positions
            std::fill(idx.begin(), idx.end(), 0);
            bool done = false;
            
            while (!done) {
                // Copy element from source to dest
                auto dest_idx = idx;
                dest_idx[axis] += offset;
                result.set(dest_idx, arr.get(idx));
                
                // Increment index
                for (int d = ndim - 1; d >= 0; --d) {
                    if (d == axis) continue;
                    if (++idx[d] < static_cast<std::size_t>(arr.shape[d])) {
                        break;
                    }
                    idx[d] = 0;
                    if (d == 0 || (d == 1 && axis == 0)) {
                        done = true;
                    }
                }
                
                // Handle axis dimension
                if (!done && axis < ndim) {
                    bool carry = true;
                    for (int d = ndim - 1; d >= 0 && carry; --d) {
                        if (d == axis) {
                            if (++idx[d] < axis_size) {
                                carry = false;
                            } else {
                                idx[d] = 0;
                            }
                        }
                    }
                    if (carry && idx[axis] == 0) {
                        done = true;
                    }
                }
            }
            
            offset += axis_size;
        }
        
        return result;
    }

    // =================================================================
    // Stack
    // Reference: numpy-reference/reference/generated/numpy.stack.html
    // =================================================================

    /**
     * @brief Join a sequence of arrays along a new axis.
     *
     * All arrays must have the same shape.
     *
     * @param arrays Sequence of arrays to stack.
     * @param axis Position where new axis is inserted (default: 0).
     * @throws std::invalid_argument if shapes are incompatible.
     */
    template <typename T>
    auto stack(const std::vector<Ndarray<T>>& arrays, int axis = 0)
        -> Ndarray<T> {
        if (arrays.empty()) {
            throw std::invalid_argument("stack: need at least one array");
        }
        
        const auto& first = arrays[0];
        const int ndim = static_cast<int>(first.ndim());
        
        // Normalize axis
        if (axis < 0) {
            axis += ndim + 1;
        }
        if (axis < 0 || axis > ndim) {
            throw std::invalid_argument("stack: axis out of bounds");
        }
        
        // Check all arrays have same shape
        for (std::size_t i = 1; i < arrays.size(); ++i) {
            if (arrays[i].shape != first.shape) {
                throw std::invalid_argument("stack: all arrays must have same shape");
            }
        }
        
        // Compute output shape (insert new dimension)
        std::vector<int> out_shape;
        out_shape.reserve(ndim + 1);
        for (int d = 0; d < axis; ++d) {
            out_shape.push_back(first.shape[d]);
        }
        out_shape.push_back(static_cast<int>(arrays.size()));
        for (int d = axis; d < ndim; ++d) {
            out_shape.push_back(first.shape[d]);
        }
        
        // Allocate output
        Ndarray<T> result(out_shape, first.type);
        
        // Copy data
        for (std::size_t i = 0; i < arrays.size(); ++i) {
            std::vector<std::size_t> idx_in(ndim, 0);
            bool done = false;
            
            while (!done) {
                // Build output index
                std::vector<std::size_t> idx_out;
                idx_out.reserve(ndim + 1);
                for (int d = 0; d < axis; ++d) {
                    idx_out.push_back(idx_in[d]);
                }
                idx_out.push_back(i);
                for (int d = axis; d < ndim; ++d) {
                    idx_out.push_back(idx_in[d]);
                }
                
                result.set(idx_out, arrays[i].get(idx_in));
                
                // Increment input index
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++idx_in[d] < static_cast<std::size_t>(first.shape[d])) {
                        break;
                    }
                    idx_in[d] = 0;
                    if (d == 0) {
                        done = true;
                    }
                }
            }
        }
        
        return result;
    }

    // =================================================================
    // Convenience stacking functions
    // Reference: numpy-reference/reference/generated/numpy.vstack.html (etc.)
    // =================================================================

    /**
     * @brief Stack arrays vertically (row-wise).
     *
     * Equivalent to concatenate(arrays, axis=0) for 2D+ arrays.
     * For 1D arrays, stacks them as rows into a 2D array.
     */
    template <typename T>
    auto vstack(const std::vector<Ndarray<T>>& arrays) -> Ndarray<T> {
        if (arrays.empty()) {
            throw std::invalid_argument("vstack: need at least one array");
        }
        
        // If 1D, reshape to (1, N) first
        std::vector<Ndarray<T>> reshaped;
        reshaped.reserve(arrays.size());
        
        for (const auto& arr : arrays) {
            if (arr.ndim() == 1) {
                reshaped.push_back(arr.reshape({1, arr.shape[0]}));
            } else {
                reshaped.push_back(arr);
            }
        }
        
        return concatenate(reshaped, 0);
    }

    /**
     * @brief Stack arrays horizontally (column-wise).
     *
     * Equivalent to concatenate(arrays, axis=1) for 2D+ arrays.
     * For 1D arrays, concatenates them into a single 1D array.
     */
    template <typename T>
    auto hstack(const std::vector<Ndarray<T>>& arrays) -> Ndarray<T> {
        if (arrays.empty()) {
            throw std::invalid_argument("hstack: need at least one array");
        }
        
        if (arrays[0].ndim() == 1) {
            return concatenate(arrays, 0);
        }
        
        return concatenate(arrays, 1);
    }

    /**
     * @brief Stack arrays depth-wise (along third axis).
     *
     * Takes a sequence of arrays and stacks them along the third axis.
     * 1D or 2D arrays are first reshaped to (M, N, 1).
     */
    template <typename T>
    auto dstack(const std::vector<Ndarray<T>>& arrays) -> Ndarray<T> {
        if (arrays.empty()) {
            throw std::invalid_argument("dstack: need at least one array");
        }
        
        std::vector<Ndarray<T>> reshaped;
        reshaped.reserve(arrays.size());
        
        for (const auto& arr : arrays) {
            if (arr.ndim() == 1) {
                reshaped.push_back(arr.reshape({1, arr.shape[0], 1}));
            } else if (arr.ndim() == 2) {
                reshaped.push_back(arr.reshape({arr.shape[0], arr.shape[1], 1}));
            } else {
                reshaped.push_back(arr);
            }
        }
        
        return concatenate(reshaped, 2);
    }

    /**
     * @brief Stack 1D arrays as columns into a 2D array.
     */
    template <typename T>
    auto column_stack(const std::vector<Ndarray<T>>& arrays) -> Ndarray<T> {
        if (arrays.empty()) {
            throw std::invalid_argument("column_stack: need at least one array");
        }
        
        // Reshape 1D arrays to (N, 1)
        std::vector<Ndarray<T>> reshaped;
        reshaped.reserve(arrays.size());
        
        for (const auto& arr : arrays) {
            if (arr.ndim() == 1) {
                reshaped.push_back(arr.reshape({arr.shape[0], 1}));
            } else if (arr.ndim() == 2) {
                reshaped.push_back(arr);
            } else {
                throw std::invalid_argument(
                    "column_stack: arrays must be 1D or 2D");
            }
        }
        
        return concatenate(reshaped, 1);
    }

    /**
     * @brief Stack 1D arrays as rows into a 2D array.
     *
     * Equivalent to vstack for 1D arrays.
     */
    template <typename T>
    auto row_stack(const std::vector<Ndarray<T>>& arrays) -> Ndarray<T> {
        return vstack(arrays);
    }

} // namespace np

#endif // NP_CONCATENATE_HPP
