#ifndef __NDARRAY_ITERATOR_H__
#define __NDARRAY_ITERATOR_H__

#include <vector>
#include <cstddef>

namespace np {
    template <typename _Tp>
    class NdArray_iterator {
    public:
        /**
         * @brief Constructs an NdArray_iterator with the given data pointer and shape.
         * 
         * @param data Pointer to the data array.
         * @param shape Vector representing the shape of the array.
         */
        NdArray_iterator(_Tp* data, const std::vector<size_t>& shape);

        /**
         * @brief Checks if there are more elements to iterate over.
         * 
         * @return true if there are more elements, false otherwise.
         */
        bool hasNext() const;

        /**
         * @brief Returns the next element in the iteration.
         * 
         * @return Reference to the next element.
         */
        _Tp& next();

        /**
         * @brief Increments the current index to point to the next element.
         */
        void _increment_index();

        /**
         * @brief Pointer to the data array.
         */
    private:
        _Tp* _data;

        /**
         * @brief Vector representing the shape of the array.
         */
        std::vector<size_t> _shape;

        /**
         * @brief Vector representing the strides of the array.
         */
        std::vector<size_t> _strides;

        /**
         * @brief Vector representing the current index in the array.
         */
        std::vector<size_t> _index;
    };
}

#endif 