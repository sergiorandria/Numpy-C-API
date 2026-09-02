/**
 * @file memristor.hpp
 * @brief Analog in-memory computing — ReRAM crossbar, Mythic/d-Matrix.
 */
#ifndef NP_MEMRISTOR_HPP
#define NP_MEMRISTOR_HPP

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

namespace np::analog
{

  struct Crossbar
  {
    ndarray<float> weights; // conductance
    Crossbar() = default;
    explicit Crossbar(ndarray<float> w) : weights(std::move(w))
    {
    }
    NP_NODISCARD ndarray<float> dot(const ndarray<float>& x) const
    {
      // O(1) analog V=IR: dot as matmul with weights^T
      auto xt = x.reshape({x.size(), 1});
      auto wt = weights.transpose();
      auto y = linalg::matmul(wt, xt);
      return y.reshape({static_cast<int>(y.size())});
    }
    NP_NODISCARD ndarray<float> quantize(int bits = 4) const
    {
      ndarray<float> q(weights.shape);
      auto& qd = q.data();
      auto& wd = weights.data();
      float scale = (1 << bits) - 1;
      for (size_t i = 0; i < wd.size(); ++i)
        qd[i] = std::round(wd[i] * scale) / scale;
      return q;
    }
  };

  struct ReRAMFactory
  {
    NP_NODISCARD static Crossbar crossbar(const ndarray<float>& w)
    {
      return Crossbar(w);
    }
  };

} // namespace np::analog

#endif // NP_MEMRISTOR_HPP
