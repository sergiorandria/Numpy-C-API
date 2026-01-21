#include "ndarray.h"

/**
 * @brief Checks if all elements along the specified axis evaluate to true.
 * 
 * @param axis The axis along which to perform the check.
 * @param out Optional output array to store the result.
 * @param keepdims If true, retains reduced dimensions with size one.
 * @param where Optional condition array to apply the check.
 * @return True if all elements evaluate to true, otherwise false.
 */

template <typename _Tp>
bool np::Ndarray<_Tp>::all(size_t axis, std::optional<Ndarray<_Tp>> out, bool keepdims, std::vector<bool> where) const
{
    return false;
}

/**
 * @brief Checks if any element along the specified axis evaluates to true.
 * 
 * @param axis The axis along which to perform the check.
 * @return True if any element evaluates to true, otherwise false.
 */

template<typename _Tp>
bool np::Ndarray<_Tp>::any(size_t axis) const 
{
    return false;
}

/**
 * @brief Returns the indices of the maximum values along the specified axis.
 * 
 * @param axis The axis along which to find the maximum values.
 * @return Indices of the maximum values.
 */

template <typename _Tp>
bool np::Ndarray<_Tp>::argmax(size_t axis) const
{
    return false;
}

/**
 * @brief Returns the indices of the minimum values along the specified axis.
 * 
 * @return Indices of the minimum values.
 */

template <typename _Tp>
bool np::Ndarray<_Tp>::argmin() const
{
    return false;
}

/**
 * @brief Performs an indirect partition along the specified axis.
 * 
 * @return Indices that would partition the array.
 */

template <typename _Tp>
int np::Ndarray<_Tp>::argpartition() const
{
    return 0;
}
