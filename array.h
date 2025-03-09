/**
 * @file Array.h
 * @brief Header file for the np::Array class template.
 *
 * This file contains the definition of the np::Array class template, which is a part of the Numpy-like library.
 * The Array class provides a container for numerical data, supporting various types and initialization methods.
 * It inherits from np::_Numpy_ndarray_internal to leverage internal ndarray functionalities.
 *
 * @note This file is part of the Numpy project.
 */

#ifndef ARRAY_H 
#define ARRAY_H 

#include <vector>
#include <list>
#include <ranges>
#include <initializer_list>

#include "dtype.h"
#include "__ndarray_internal.h"

namespace np {
    /**
     * @class Array
     * @brief A template class for creating and managing numerical arrays.
     *
     * The Array class template provides a flexible container for numerical data, supporting initialization
     * from various container types and initializer lists. It inherits from np::_Numpy_ndarray_internal to
     * utilize internal ndarray functionalities.
     *
     * @tparam _Tp The type of elements stored in the array.
     */
    template <typename _Tp>
    class Array: public np::_Numpy_ndarray_internal<_Tp>
    {
    public: 
        /**
         * @brief Default constructor for the Array class.
         */
        Array () = default;
        
        /**
         * @brief Constructs an Array from a given container.
         *
         * This constructor initializes the Array with elements from the provided container.
         *
         * @tparam Container The type of the container.
         * @param container The container from which to initialize the Array.
         * @param type Optional parameter to specify the data type of the Array elements. Defaults to np::int32.
         */
        template<typename Container>
            requires std::ranges::range<Container>
        Array(const Container& container, std::optional<np::dtype> type = np::int32);
        
        /**
         * @brief Constructs an Array from an initializer list.
         *
         * This constructor initializes the Array with elements from the provided initializer list.
         *
         * @param initList The initializer list from which to initialize the Array.
         * @param type Optional parameter to specify the data type of the Array elements. Defaults to np::int32.
         */
        Array(std::initializer_list<_Tp> initList, std::optional<np::dtype> type = np::int32);
        
        /**
         * @brief Default destructor for the Array class.
         */
        ~Array() = default; 
        
        /**
         * @brief Accesses the element at the specified index.
         *
         * This operator provides access to the element at the given index in the Array.
         *
         * @param i The index of the element to access.
         * @return A reference to the element at the specified index.
         */
        _Tp& operator[](int i);

        /**
         * @brief Move assignment operator for the Array class.
         *
         * This operator assigns the contents of the given Array to the current Array using move semantics.
         *
         * @param array The Array to move.
         * @return A reference to the current Array.
         */
        Array& operator=(const Array&& array);
    };;
}

#include "array.tpp"
#include "_array_creation_routine.tpp"

#endif 