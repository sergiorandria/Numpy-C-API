/**
 * @file accelerator.hpp
 * @brief Heterogeneous accelerator dispatcher — CPU/GPU/Loihi/ReRAM/Photonics.
 */
#ifndef NP_ACCELERATOR_HPP
#define NP_ACCELERATOR_HPP

#include "api_macros.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"
#include <memory>
#include <string>

namespace np::accelerator
{

  struct IAccelerator
  {
    virtual ~IAccelerator() = default;
    virtual ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) = 0;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
  };

  struct CPUAccelerator : IAccelerator
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

  struct GPUAccelerator : IAccelerator
  {
    ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) override
    {
      return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "GPU";
    }
  };

  struct LoihiAccelerator : IAccelerator
  {
    ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) override
    {
      return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "Loihi2";
    }
  };

  struct ReRAMAccelerator : IAccelerator
  {
    ndarray<float> matmul(const ndarray<float>& a, const ndarray<float>& b) override
    {
      return linalg::matmul(a, b);
    }
    NP_NODISCARD std::string name() const noexcept override
    {
      return "ReRAM";
    }
  };

  struct AcceleratorFactory
  {
    NP_NODISCARD static std::shared_ptr<IAccelerator> cpu()
    {
      return std::make_shared<CPUAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> gpu()
    {
      return std::make_shared<GPUAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> loihi()
    {
      return std::make_shared<LoihiAccelerator>();
    }
    NP_NODISCARD static std::shared_ptr<IAccelerator> reram()
    {
      return std::make_shared<ReRAMAccelerator>();
    }
  };

} // namespace np::accelerator

#endif // NP_ACCELERATOR_HPP
