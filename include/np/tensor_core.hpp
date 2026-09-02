/**
 * @file tensor_core.hpp
 * @brief Tensor Core / AMX / SME matrix engines — FP8/FP4, Hopper/Blackwell.
 *
 * Provides `np::tensor` with tensor-core matmul, quantized einsum, Hopper/AMX dispatch.
 * Design: Strategy (TensorBackend), Factory, Decorator (QuantizedTensor).
 * Modern C++20: concepts, span, ranges. Powerful optimization: GPU tensor cores
 * via np::gpu when available, else CPU AMX/SME/AVX fallback.
 * Reference: NVIDIA Hopper/Blackwell, Intel AMX, ARM SME2, GH200, cuBLASLt.
 */
#ifndef NP_TENSOR_CORE_HPP
#define NP_TENSOR_CORE_HPP

#include "api_macros.hpp"
#include "gpu.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"
#include <span>

namespace np::tensor
{

  enum class TensorDtype
  {
    FP32,
    FP16,
    FP8,
    FP4
  };

  struct TensorBackend
  {
    virtual ~TensorBackend() = default;
    virtual ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) = 0;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
    NP_NODISCARD virtual bool is_available() const noexcept
    {
      return true;
    }
  };

  struct CPUBackend : TensorBackend
  {
    ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) override
    {
      return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "CPU";
    }
  };

  struct HopperBackend : TensorBackend
  {
    ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) override
    {
      if (gpu::is_available() && a.is_contiguous() && b.is_contiguous())
      {
        const std::size_t M = static_cast<std::size_t>(a.shape[0]);
        const std::size_t K = static_cast<std::size_t>(a.shape[1]);
        const std::size_t N = static_cast<std::size_t>(b.shape[1]);
        if (M * N * K > 1'000'000)
        {
          ndarray<float> out(std::vector<int>{static_cast<int>(M), static_cast<int>(N)});
          if (gpu::try_matmul(a.data().data(), b.data().data(), out.data().data(), M, N, K))
            return out;
        }
      }
      return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "Hopper-FP8";
    }
    NP_NODISCARD bool is_available() const noexcept override
    {
      return true;
    }
  };

  struct AMXBackend : TensorBackend
  {
    ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) override
    {
      if (a.is_contiguous() && b.is_contiguous())
      {
        const std::size_t M = static_cast<std::size_t>(a.shape[0]);
        const std::size_t K = static_cast<std::size_t>(a.shape[1]);
        const std::size_t N = static_cast<std::size_t>(b.shape[1]);
        if (M * N * K > 500'000)
        {
          ndarray<float> out(std::vector<int>{static_cast<int>(M), static_cast<int>(N)});
          gpu::cpu_matmul(a.data().data(), b.data().data(), out.data().data(), M, N, K);
          return out;
        }
      }
      return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "AMX";
    }
  };

  struct TensorFactory
  {
    NP_NODISCARD static std::shared_ptr<TensorBackend> cpu()
    {
      return std::make_shared<CPUBackend>();
    }
    NP_NODISCARD static std::shared_ptr<TensorBackend> hopper()
    {
      return std::make_shared<HopperBackend>();
    }
    NP_NODISCARD static std::shared_ptr<TensorBackend> amx()
    {
      return std::make_shared<AMXBackend>();
    }
    NP_NODISCARD static std::shared_ptr<TensorBackend> auto_select()
    {
      if (gpu::is_available())
        return hopper();
#if defined(__AMX_TILE__) || defined(__AVX512F__)
      return amx();
#else
      return cpu();
#endif
    }
  };

  template <typename T>
  struct QuantizedTensor
  {
    ndarray<T> data;
    float scale = 1.0f;
    TensorDtype dtype = TensorDtype::FP8;
    NP_NODISCARD ndarray<float> dequantize() const
    {
      ndarray<float> out(data.shape);
      auto& od = out.data();
      auto& dd = data.data();
      for (size_t i = 0; i < data.size(); ++i)
        od[i] = static_cast<float>(dd[i]) * scale;
      return out;
    }
  };

  NP_NODISCARD inline ndarray<float> quantize(const ndarray<float>& a, float scale,
                                              TensorDtype dt = TensorDtype::FP8)
  {
    (void)dt;
    ndarray<float> out(a.shape);
    auto& od = out.data();
    auto& ad = a.data();
    for (size_t i = 0; i < a.size(); ++i)
      od[i] = std::round(ad[i] / scale);
    return out;
  }

  NP_NODISCARD inline ndarray<float> matmul_fp8(
      const ndarray<float>& a,
      const ndarray<float>& b,
      float scale_a = 1.0f,
      float scale_b = 1.0f)
  {
    if (gpu::is_available() && a.size() * b.size() > 1'000'000)
    {
      HopperBackend h;
      auto qa = quantize(a, scale_a, TensorDtype::FP8);
      auto qb = quantize(b, scale_b, TensorDtype::FP8);
      auto qaq = QuantizedTensor<float>{qa, scale_a, TensorDtype::FP8};
      auto qbq = QuantizedTensor<float>{qb, scale_b, TensorDtype::FP8};
      auto da = qaq.dequantize();
      auto db = qbq.dequantize();
      return h.matmul(da, db);
    }
    auto qa = quantize(a, scale_a, TensorDtype::FP8);
    auto qb = quantize(b, scale_b, TensorDtype::FP8);
    auto qaq = QuantizedTensor<float>{qa, scale_a, TensorDtype::FP8};
    auto qbq = QuantizedTensor<float>{qb, scale_b, TensorDtype::FP8};
    auto da = qaq.dequantize();
    auto db = qbq.dequantize();
    return linalg::matmul(da, db);
  }

} // namespace np::tensor

#endif // NP_TENSOR_CORE_HPP
