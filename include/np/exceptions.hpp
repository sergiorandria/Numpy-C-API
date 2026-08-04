/**
 * @file exceptions.hpp
 * @brief NumPy-compatible exception and warning types.
 *
 * All exceptions carry the source location (file, line, function) of the
 * throw site, mirroring the information density of Python tracebacks.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_EXCEPTIONS_HPP
#define NP_EXCEPTIONS_HPP

#include <exception>
#include <format>
#include <source_location>
#include <string>

namespace np::exceptions {

    namespace detail {
        [[nodiscard]] inline std::string format_message(
            const std::string& msg,
            const std::source_location& loc) {
            return std::format("{}:{}: {}: {}",
                               loc.file_name(), loc.line(),
                               loc.function_name(), msg);
        }
    } // namespace detail

    /**
     * @brief Base class for all np exceptions.
     */
    class NumpyError : public std::exception {
      public:
        explicit NumpyError(const std::string& msg,
                            const std::source_location& loc =
                                std::source_location::current())
            : what_msg_(detail::format_message(msg, loc)) {}

        [[nodiscard]] auto what() const noexcept -> const char* override {
            return what_msg_.c_str();
        }

      protected:
        std::string what_msg_;
    };

    /**
     * @brief Raised when an axis parameter is out of bounds for the array.
     */
    class AxisError : public NumpyError {
      public:
        explicit AxisError(const std::string& msg,
                           const std::source_location& loc =
                               std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    /**
     * @brief Non-fatal warning emitted for invalid axis/rank usage.
     */
    class RankWarning : public NumpyError {
      public:
        explicit RankWarning(const std::string& msg,
                             const std::source_location& loc =
                                 std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    /**
     * @brief Raised when matrix dimensions are incompatible.
     */
    class MatrixDimError : public NumpyError {
      public:
        explicit MatrixDimError(const std::string& msg,
                                const std::source_location& loc =
                                    std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    /**
     * @brief Raised when two dtypes cannot be promoted.
     */
    class DtypePromotionError : public NumpyError {
      public:
        explicit DtypePromotionError(const std::string& msg,
                                     const std::source_location& loc =
                                         std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    /**
     * @brief Emitted when an API enters a deprecated code path.
     */
    class VisibleDeprecation : public NumpyError {
      public:
        explicit VisibleDeprecation(const std::string& msg,
                                    const std::source_location& loc =
                                        std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    /**
     * @brief Emitted when a complex value is implicitly cast to real.
     */
    class ComplexWarning : public NumpyError {
      public:
        explicit ComplexWarning(const std::string& msg,
                                const std::source_location& loc =
                                    std::source_location::current())
            : NumpyError(msg, loc) {}
    };

} // namespace np::exceptions

namespace np {
    using AxisError         = exceptions::AxisError;
    using RankWarning       = exceptions::RankWarning;
    using MatrixDimError    = exceptions::MatrixDimError;
    using DtypePromotionError = exceptions::DtypePromotionError;
    using VisibleDeprecation = exceptions::VisibleDeprecation;
    using ComplexWarning    = exceptions::ComplexWarning;
} // namespace np

#endif // NP_EXCEPTIONS_HPP
