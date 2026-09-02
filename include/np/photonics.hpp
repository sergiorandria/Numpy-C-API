/**
 * @file photonics.hpp
 * @brief Photonics — Mach-Zehnder mesh, optical FFT.
 */
#ifndef NP_PHOTONICS_HPP
#define NP_PHOTONICS_HPP

#include "api_macros.hpp"
#include "ndarray.hpp"
#include <complex>

namespace np::photonics
{

  using c64 = std::complex<float>;
  using c128 = std::complex<double>;

  struct MachZehnderMesh
  {
    ndarray<c128> unitary; // 2x2 or NxN
    MachZehnderMesh() = default;
    explicit MachZehnderMesh(ndarray<c128> u) : unitary(std::move(u))
    {
    }
    NP_NODISCARD ndarray<c128> apply(const ndarray<c128>& x) const
    {
      // unitary * x (matmul)
      auto y = linalg::matmul(unitary, x.reshape({static_cast<int>(x.size()), 1}));
      return y.reshape({static_cast<int>(x.size())});
    }
  };

  struct PhotonicsFactory
  {
    NP_NODISCARD static MachZehnderMesh identity(int n)
    {
      ndarray<c128> u(std::vector<int>{n, n});
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          u(i, j) = (i == j ? c128(1, 0) : c128(0, 0));
      return MachZehnderMesh(u);
    }
  };

} // namespace np::photonics

#endif // NP_PHOTONICS_HPP
