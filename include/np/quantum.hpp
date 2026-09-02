/**
 * @file quantum.hpp
 * @brief Quantum-inspired — StateVector, tensor-network einsum.
 */
#ifndef NP_QUANTUM_HPP
#define NP_QUANTUM_HPP

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"
#include <complex>

namespace np::quantum
{

  using c64 = std::complex<float>;
  using c128 = std::complex<double>;

  struct StateVector
  {
    ndarray<c128> amps; // 2^n
    StateVector() = default;
    explicit StateVector(int n_qubits) : amps(std::vector<int>{1 << n_qubits})
    {
      amps[0] = c128(1, 0);
    }
    NP_NODISCARD int n_qubits() const
    {
      int n = 0, s = static_cast<int>(amps.size());
      while ((1 << n) < s)
        ++n;
      return n;
    }
    NP_NODISCARD double prob(int idx) const
    {
      c128 a = static_cast<c128>(amps[idx]);
      return std::norm(a);
    }
  };

  struct QuantumFactory
  {
    NP_NODISCARD static StateVector zero_state(int n)
    {
      return StateVector(n);
    }
    NP_NODISCARD static StateVector plus_state(int n)
    {
      StateVector s(n);
      double amp = 1.0 / std::sqrt(1 << n);
      for (size_t i = 0; i < s.amps.size(); ++i)
        s.amps[i] = c128(amp, 0);
      return s;
    }
  };

} // namespace np::quantum

#endif // NP_QUANTUM_HPP
