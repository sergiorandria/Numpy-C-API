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
        // Prefix a message with its throw site, formatted like a Python traceback line.
        [[nodiscard]] inline std::string format_message(
            const std::string& msg,
            const std::source_location& loc) {
            return std::format("{}:{}: {}: {}",
                               loc.file_name(), loc.line(),
                               loc.function_name(), msg);
        }
    } // namespace detail

    // Base class for all np exceptions; what() carries "file:line: function: msg".
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

    // Raised when an axis parameter is out of bounds for the array.
    // Reference: numpy-reference/reference/generated/numpy.exceptions.AxisError.html
    class AxisError : public NumpyError {
      public:
        explicit AxisError(const std::string& msg,
                           const std::source_location& loc =
                               std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    // Non-fatal warning emitted for invalid axis/rank usage.
    // Reference: numpy-reference/reference/generated/numpy.RankWarning.html
    class RankWarning : public NumpyError {
      public:
        explicit RankWarning(const std::string& msg,
                             const std::source_location& loc =
                                 std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    // Raised when matrix dimensions are incompatible.
    class MatrixDimError : public NumpyError {
      public:
        explicit MatrixDimError(const std::string& msg,
                                const std::source_location& loc =
                                    std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    // Raised when two dtypes cannot be promoted.
    class DtypePromotionError : public NumpyError {
      public:
        explicit DtypePromotionError(const std::string& msg,
                                     const std::source_location& loc =
                                         std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    // Emitted when an API enters a deprecated code path.
    class VisibleDeprecation : public NumpyError {
      public:
        explicit VisibleDeprecation(const std::string& msg,
                                    const std::source_location& loc =
                                        std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    // Emitted when a complex value is implicitly cast to real.
    class ComplexWarning : public NumpyError {
      public:
        explicit ComplexWarning(const std::string& msg,
                                const std::source_location& loc =
                                    std::source_location::current())
            : NumpyError(msg, loc) {}
    };

    // Raised by np::linalg when a decomposition fails to converge or
    // encounters a mathematically singular problem.
    // Reference: numpy-reference/reference/generated/numpy.linalg.LinAlgError.html
    class LinAlgError : public NumpyError {
      public:
        explicit LinAlgError(const std::string& msg,
                             const std::source_location& loc =
                                 std::source_location::current())
            : NumpyError(msg, loc) {}
    };

} // namespace np::exceptions

namespace np {
    using AxisError           = exceptions::AxisError;
    using RankWarning         = exceptions::RankWarning;
    using MatrixDimError      = exceptions::MatrixDimError;
    using DtypePromotionError = exceptions::DtypePromotionError;
    using VisibleDeprecation  = exceptions::VisibleDeprecation;
    using ComplexWarning      = exceptions::ComplexWarning;
    using LinAlgError         = exceptions::LinAlgError;
} // namespace np

#endif // NP_EXCEPTIONS_HPP
