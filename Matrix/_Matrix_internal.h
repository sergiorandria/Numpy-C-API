#ifndef _MATRIX_INTERNAL
#define _MATRIX_INTERNAL
/*
    numpy::Matrix:
        returns a matrix from an array-like object,
        or from a string data
*/

#include <optional>
#include <string>
#include <vector>
#include <string_view>

#include "../dtype.hpp"
#include "../ndarray.hpp"

namespace np
{
    template<typename Tp>
    class MatrixInternal: public np::Ndarray<Tp>
    {
      public:
        virtual ~MatrixInternal() = 0;

        virtual auto operator*(const MatrixInternal<Tp> other)
        -> MatrixInternal<Tp>;
        virtual auto operator+(const MatrixInternal<Tp> other)
        -> MatrixInternal<Tp>;
        virtual auto operator-(const MatrixInternal<Tp> other)
        -> MatrixInternal<Tp>;
        virtual auto operator*(const double value)
        -> MatrixInternal<Tp>;

      protected:
        np::dtype type;      // data type of the output matrix
        std::bool_constant<false>
        copy;     //determines whether the data is copied or whether a view is constructed
    };
}  // namespace np

#endif