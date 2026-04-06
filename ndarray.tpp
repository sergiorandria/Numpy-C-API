#ifndef __NDARRAY_TPP
#define __NDARRAY_TPP

#include <cstdint>
#include <iostream>
#include <ranges>
#include <utility>

#include "__dtype_internal.h"
#include "ndarray.hpp"
#include "__dtype_conversion.h"

namespace np {
    static constexpr int LOOP_UNROLLING_MAX_BOUND = 16;
    namespace {
        enum NP_DTYPE_CCLASS_SIZE : std::uint8_t
        {
            // Just let the magic works,
            // These magic number are not magic at all
            INT16_SZ    = sizeof( std::int16_t ),
            INT32_SZ    = sizeof( std::int32_t ),
            INT64_SZ    = sizeof( std::int64_t ),
            UINT8_SZ    = sizeof( std::uint8_t ),
            UINT16_SZ   = sizeof( std::uint16_t ),
            UINT32_SZ   = sizeof( std::uint32_t ),
            UINT64_SZ   = sizeof( std::uint64_t )
            
        };
    } // namespace
    
    namespace {
        // Helper to detect initializer_list
        template<typename T>
        struct is_initializer_list_v : std::false_type {};
        
        template<typename T>
        struct is_initializer_list_v<std::initializer_list<T>> : std::true_type {};
        
        // Get the underlying element type
        template<typename T>
        struct underlying_element_type
        {
            using type = T;
        };
        
        template<typename T>
        struct underlying_element_type<std::initializer_list<T>>
        {
            using type = typename underlying_element_type<T>::type;
        };
        
        // Recursive shape extractor - base case for non-initializer_list
        template<typename T>
        auto extract_shape_impl( const T &,
        std::true_type ) -> decltype( std::vector<int> {} )
        
        {
            return {};  // Base case: no more dimensions
        }
        
        template<typename T>
        std::vector<int> extract_shape_impl( const std::initializer_list<T> &list )
        {
            std::vector<int> shape;
            shape.push_back( static_cast<int>( list.size() ) );
            
            if ( list.size() > 0 )
            {
                // Recurse into the first element (all sub‑lists have the same shape)
                shape.insert( shape.end(), extract_shape_impl( *list.begin() ).begin(),
                              extract_shape_impl( *list.begin() ).end() );
            }
            
            return shape;
        }
        
        // Public interface: extracts shape from a nested initializer_list
        template<typename T>
        std::vector<int> extract_shape( const std::initializer_list<T> &list )
        {
            if constexpr( is_initializer_list_v<T>::value )
            {
                return extract_shape_impl( list );
            }
            
            else
            {
                // Base case: flat list of values – shape is simply its size
                return {static_cast<int>( list.size() )};
            }
        }
        
        // Validate that all sub-lists have consistent sizes
        template<typename T>
        auto validate_shape( const std::initializer_list<T> &list,
                             const std::vector<int> &expected_shape,
        size_t depth = 0 ) -> decltype( bool {} )
        
        {
            if ( depth >= expected_shape.size() )
            {
                return true;
            }
            
            if ( list.size() != static_cast<size_t>( expected_shape[depth] ) )
            {
                return false;
            }
            
            if constexpr( is_initializer_list_v<T>::value )
            {
                for ( const auto& sublist : list )
                {
                    if ( !validate_shape( sublist, expected_shape, depth + 1 ) )
                    {
                        return false;
                    }
                }
            }
            
            return true;
        }
        
        // Flatten nested initializer_list into a vector
        template<typename T>
        void flatten_g( const std::initializer_list<T> &list, std::vector<T> &out )
        {
            for ( const auto& elem : list )
            {
                out.push_back( elem );
            }
        }
        
        template<typename T>
        void flatten( const std::initializer_list<std::initializer_list<T>> &list,
                      std::vector<T> &out )
        {
            for ( const auto& sublist : list )
            {
                flatten_g( sublist, out );
            }
        }
        
        template<typename T>
        void flatten_g( const
                        std::initializer_list<std::initializer_list<std::initializer_list<T>>> &list,
                        std::vector<T> &out )
        {
            for ( const auto& sublist : list )
            {
                flatten_g( sublist, out );
            }
        }
    } // namespace
    
    /* template <typename T>
    template <typename Container>
    requires std::ranges::range<Container>
    constexpr np::Ndarray<T>::Ndarray(
        const std::optional<Container> &container ) : order( matrix::Order::C )
    {
        static_assert( !container.empty(),
                       "Cannot construct Ndarray object with empty container" );
        np::dtype type = cxx_to_np_type<decltype( container[0] )>();
        this->type = type;
        // Should continue moving element into std::optional<std::vector<data>>
    }
     */
    /**
        @brief Constructs an Ndarray object with the specified shape, data type, and
        other optional parameters.
    
        @tparam _Tp The type of the elements in the Ndarray.
        @param shape A tuple representing the shape of the Ndarray.
        @param type The data type of the elements in the Ndarray.
        @param buffer An optional parameter that provides a buffer to initialize the
        Ndarray.
        @param offset An optional parameter that specifies the offset in the buffer.
        @param strides An optional parameter that specifies the strides for each
        dimension.
        @param order The memory layout order of the Ndarray (C or Fortran order).
    */
    template<typename Tp>
    constexpr np::Ndarray<Tp>::Ndarray( std::initializer_list<Tp> initList )
        : order( matrix::Order::C ),
          type( cxx_to_np_type<Tp>::value ),
          shape( initList.begin(), initList.end() )
    {
        // Validate shape consistency for nested lists
        if constexpr( is_initializer_list_v<Tp>::value )
        {
            if ( !validate_shape( initList, shape, 0 ) )
            {
                throw std::runtime_error( "Inconsistent shape in nested initializer_list" );
            }
        }
        
        // Compute total size
        size_t total_size = 1;
        
        for ( int dim : shape )  // No pragma unroll - runtime loop
        {
            total_size *= dim;
        }
        
        // Allocate and fill buffer
        buffer = std::vector<Tp>( total_size );
        // Compute strides
        strides = _compute_strides();
        this->ndim = shape.size();
        this->itemsize = sizeof( Tp );
    }
    
    // Fixed container constructor
    template<typename Tp>
    template<typename Container>
    requires std::ranges::range<Container>
    constexpr np::Ndarray<Tp>::Ndarray( const Container&
                                        container ) // Removed optional
        : order( matrix::Order::C ),
          type( cxx_to_np_type<typename Container::value_type>::value )
    {
        if ( container.empty() )
        {
            throw std::invalid_argument( "Cannot construct Ndarray with empty container" );
        }
        
        // Handle container as 1D array
        shape = {static_cast<int>( container.size() )};
        this->ndim = 1;
        this->itemsize = sizeof( Tp );
        buffer = std::vector<Tp>( container.begin(), container.end() );
        strides = _compute_strides();
    }
    
    // If you really need optional, do this instead:
    template<typename Tp>
    template<typename Container>
    requires std::ranges::range<Container>
    constexpr np::Ndarray<Tp>::Ndarray( const std::optional<Container>
                                        &opt_container )
        : order( matrix::Order::C )
    {
        if ( !opt_container.has_value() || opt_container->empty() )
        {
            throw std::invalid_argument( "Optional container is empty or has no value" );
        }
        
        const auto& container = *opt_container;
        type = cxx_to_np_type<typename Container::value_type>::value;
        shape = {static_cast<int>( container.size() )};
        this->ndim = 1;
        this->itemsize = sizeof( Tp );
        buffer = std::vector<Tp>( container.begin(), container.end() );
        strides = _compute_strides();
    }
    
    template<typename Tp>
    constexpr np::Ndarray<Tp>::Ndarray( const std::vector<int> &shape,
                                        np::dtype type,
                                        std::optional<std::vector<Tp>> buffer,
                                        std::optional<off_t> offset,
                                        std::optional < std::vector<size_t>> strides,
                                        np::matrix::Order order ) noexcept : type( type ), shape( shape ),
        buffer( buffer ),
        offset( offset ), strides( *strides ), order( order )
    {
        // Assign the memory layout order
        switch ( type )
        {
            case np::int8:
            case np::int16:
                {
                    this->itemsize = INT16_SZ;
                }
                break;
                
            case np::int32:
                {
                    this->itemsize = INT32_SZ;
                }
                break;
                
            case np::int64:
                {
                    this->itemsize = INT64_SZ;
                }
                break;
                
            case np::uint8:
                {
                    this->itemsize = UINT8_SZ;
                }
                break;
                
            case np::uint16:
                {
                    this->itemsize = UINT16_SZ;
                }
                break;
                
            case np::uint32:
                {
                    this->itemsize = UINT32_SZ;
                }
                break;
                
            case np::uint64:
                {
                    this->itemsize = UINT64_SZ;
                }
                break;
                
            case np::float16:
            case np::float32:
            case np::float64:
            case np::longdouble:
            case np::complex64:
            case np::complex128:
            case np::clongdouble:
            case np::bool_:
            case np::string_:
            case np::unicode_:
            case np::datetime64:
            case np::timedelta64:
            case np::void_:
            case np::object_:
                {
                }
                break;
        }
        
        #if defined(COMPILE_TIME_INSTR)
        auto size = std::tuple_size < decltype( this->shape ) >::value;
        
        if ( size != 0 )
        {
            for ( int i = 0; i != size; ++i )
            {
                _Tp value = std::get<i>( this->shape );
            }
        }
        
        else
        {
            throw std::invalid_argument( "Ndarray" );
        }
        
        #endif
        size_t _ts = 1;
        // To optimize compilation time
        // as much as we can.
#pragma unroll LOOP_UNROLLING_MAX_BOUND
        
        for ( int dim : shape )
        {
            _ts *= dim;
        }
        
        // Allocate or use provided buffer
        if ( buffer.has_value() )
        {
            this->buffer = buffer;
            
            if ( this->buffer->size() != _ts )
            {
                this->buffer->resize( _ts );
            }
        }
        
        else
        {
            this->buffer = std::vector<Tp>( _ts, Tp() );
        }
        
        // Compute strides (ignoring optional parameter for simplicity)
        this->strides = _compute_strides();
    }
    
    template <typename Tp>
    auto np::Ndarray<Tp>::_compute_strides() const -> decltype(
        std::vector<size_t>() )
    {
        std::vector<size_t> computed_strides( this->shape.size(), 1 );
        size_t stride = 1;
#pragma unroll LOOP_UNROLLING_MAX_BOUND
        
        for ( int i = this->shape.size() - 1; i >= 0; --i )
        {
            computed_strides[i] = stride;
            stride *= this->shape[i];
        }
        
        return computed_strides;
    }
    
    template <typename Tp>
    template <std::size_t Size>
    auto np::Ndarray<Tp>::_get_flat_index( const
    std::array<std::size_t, Size> &indices ) const -> decltype( std::size_t {} )
    
    {
        std::size_t flatIndex = 0;
#pragma unroll LOOP_UNROLLING_MAX_BOUND
        
        for ( std::size_t i = 0; i < Size; i++ )
        {
            flatIndex += indices[i] * this->strides[i];
        }
        
        return flatIndex;
    }
    
    template<typename T>
    template<std::size_t Size>
    void np::Ndarray<T>::set( const std::array<std::size_t, Size> &indices,
                              T value )
    {
        if ( !buffer.has_value() )
        {
            // Should not happen if constructor allocated buffer, but guard anyway
            size_t total_size = 1;
#pragma unroll LOOP_UNROLLING_MAX_BOUND
            
            for ( int dim : shape )
            {
                total_size *= dim;
            }
            
            buffer = std::vector<T>( total_size, T() );
        }
        
        std::size_t idx = _get_flat_index( indices );
        ( *buffer )[idx] = value;
    }
    
    template <typename Tp>
    void np::Ndarray<Tp>::_print_recursive( std::size_t dim, std::size_t offset,
                                            std::ostream & output ) const
    {
        if ( !buffer.has_value() )
        {
            output << "[]";
            return;
        }
        
        if ( dim == this->shape.size() - 1 )
        {
            output << "[";
#pragma unroll LOOP_UNROLLING_MAX_BOUND
            
            for ( std::size_t i = 0; i < this->shape[dim]; i++ )
            {
                if ( i != 0 )
                {
                    output << ",";
                }
                
                output << ( *buffer )[offset + i];
            }
            
            output << "]";
        }
        
        else
        {
            output << "[";
#pragma unroll LOOP_UNROLLING_MAX_BOUND
            
            for ( std::size_t i = 0; i < this->shape[dim]; i++ )
            {
                if ( i != 0 )
                {
                    output << ",";
                }
                
                _print_recursive( dim + 1, offset + ( i * strides[dim] ), output );
            }
            
            output << "]";
        }
    }
    
    template<typename Tp>
    template<typename Container>
    void np::Ndarray<Tp>::set( const Container & indices, Tp value )
    {
        if ( !buffer.has_value() )
        {
            // Allocate buffer if not allocated
            size_t total_size = 1;
#pragma unroll LOOP_UNROLLING_MAX_BOUND
            
            for ( int dim : shape )
            {
                total_size *= dim;
            }
            
            buffer = std::vector<Tp>( total_size, Tp() );
        }
        
        // Calculate flat index
        size_t flat_index = 0;
#pragma unroll LOOP_UNROLLING_MAX_BOUND
        
        for ( size_t i = 0; i < indices.size(); ++i )
        {
            flat_index += indices[i] * strides[i];
        }
        
        // Set the value
        ( *buffer )[flat_index] = value;
    }
    
    template<typename T>
    template<std::size_t Size>
    auto np::Ndarray<T>::get( const std::array<std::size_t, Size> &indices ) ->
    T &
    {
        if ( !buffer.has_value() )
        {
            throw std::runtime_error( "No buffer" );
        }
        
        return ( *buffer )[_get_flat_index( indices )];
    }
    
    template<typename T>
    template<std::size_t Size>
    auto np::Ndarray<T>::get( const std::array<std::size_t, Size> &indices )
    const -> const T &
    {
        if ( !buffer.has_value() )
        {
            throw std::runtime_error( "No buffer" );
        }
        
        return ( *buffer )[_get_flat_index( indices )];
    }
    
    // Get implementation
    template<typename Tp>
    template<typename Container>
    auto np::Ndarray<Tp>::get( const Container & indices ) const -> decltype(
    Tp {} )
    
    {
        if ( !buffer.has_value() )
        {
            return Tp();
        }
        
        // Calculate flat index
        size_t flat_index = 0;
#pragma unroll LOOP_UNROLLING_MAX_BOUND
        
        for ( size_t i = 0; i < indices.size(); ++i )
        {
            flat_index += indices[i] * strides[i];
        }
        
        return ( *buffer )[flat_index];
    }
    
    /**
        @brief Constructs an Ndarray from a nested initializer list (N-dimensional)
    
        This constructor recursively processes nested initializer lists to create
        an N-dimensional Ndarray. All sub-lists at each dimension must have
        consistent sizes.
    
        @tparam T The data type of elements in the array
        @param nested_list A recursively nested initializer list structure
    
        @throws std::invalid_argument If dimensions are inconsistent or list is empty
    
        @example
        @code
        // 1D array
        np::Ndarray<int> a = {1, 2, 3, 4};
    
        // 2D array (2x3)
        np::Ndarray<int> b = {{1, 2, 3}, {4, 5, 6}};
    
        // 3D array (2x2x3)
        np::Ndarray<int> c = {{{1,2,3},{4,5,6}}, {{7,8,9},{10,11,12}}};
        @endcode
    */
    template<typename T>
    constexpr Ndarray<T>::Ndarray( std::initializer_list<std::initializer_list<T>>
                                   nested_list )
        : order( matrix::Order::C )
    {
        // This constructor handles 2D case specifically
        // For N-dimensional, you'd need recursive template metaprogramming
        _construct_from_nested( nested_list );
    }
    
    /**
        @brief Recursively constructs N-dimensional array from nested initializer
        lists
    
        @tparam U The current level's initializer list type
        @param list The initializer list at current depth
        @param depth Current recursion depth (0-based)
    */
    template <typename T>
    template<typename U>
    void np::Ndarray<T>::_construct_from_nested( std::initializer_list<U> list,
            size_t depth )
    {
        if ( depth == 0 )
        {
            shape.clear();
            buffer.clear();
        }
        
        if constexpr( std::is_same_v<U, T> )
        {
#pragma unroll LOOP_UNROLLING_MAX_BOUND
        
            for ( const auto& elem : list )
            {
                buffer.push_back( elem );
            }
            
            if ( depth == 1 )
            {
                shape.push_back( list.size() );
            }
        }
        
        else
        {
            // Recursive case: still in nested lists
            shape.push_back( list.size() );
            size_t expected_size = 0;
            bool first = true;
#pragma unroll LOOP_UNROLLING_MAX_BOUND
            
            for ( const auto& inner : list )
            {
                size_t current_size = inner.size();
                
                if ( first )
                {
                    expected_size = current_size;
                    first = false;
                }
                
                else
                    if ( current_size != expected_size )
                    {
                        throw std::invalid_argument( "Inconsistent dimensions in nested initializer list" );
                    }
                    
                _construct_from_nested( inner, depth + 1 );
            }
        }
        
        this->ndim = shape.size();
        this->itemsize = sizeof( T );
        this->strides = _compute_strides();
        this->type = cxx_to_np_type<T>::value;
    }
} // namespace np

#endif