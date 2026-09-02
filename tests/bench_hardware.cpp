/**
 * @file bench_hardware.cpp
 * @brief Micro-benchmark for hardware backends — HBM, tensor, neuromorphic, padic,
 * lattice.
 *
 * Measures throughput for:
 *   1. HBM migrate (mem::migrate_to_hbm)
 *   2. Tensor FP8 matmul (tensor::matmul_fp8)
 *   3. ReRAM crossbar dot (analog::Crossbar)
 *   4. Photonics mesh apply
 *   5. Neuromorphic spike encode + LIF
 *   6. Padic Hensel lift
 *   7. Lattice LLL
 *
 * Not part of ctest; run: ./build/tests/bench_hardware
 */
#include <np/np.hpp>
#include <chrono>
#include <cstdio>

using Clock = std::chrono::steady_clock;
template <typename Fn>
double ms(Fn&& fn, int iters = 3)
{
  double best = 1e18;
  for (int i = 0; i < iters; ++i)
  {
    auto t0 = Clock::now();
    fn();
    auto t1 = Clock::now();
    double d = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (d < best)
      best = d;
  }
  return best;
}

int main()
{
  auto a = np::eye<float>(64);
  auto b = np::eye<float>(64);
  printf(
      "HBM migrate: %.2f ms\n",
      ms(
          [&]
          {
            auto h = np::mem::migrate_to_hbm(a);
            (void)h;
          }));
  printf(
      "tensor matmul_fp8: %.2f ms\n",
      ms(
          [&]
          {
            auto c = np::tensor::matmul_fp8(a, b);
            (void)c;
          }));
  printf(
      "ReRAM dot: %.2f ms\n",
      ms(
          [&]
          {
            np::analog::Crossbar cb(a);
            auto x = np::ndarray<float>(std::vector<int>{64});
            for (int i = 0; i < 64; ++i)
              x[i] = 1.0f;
            auto y = cb.dot(x);
            (void)y;
          }));
  printf(
      "photonics: %.2f ms\n",
      ms(
          [&]
          {
            auto mesh = np::photonics::PhotonicsFactory::identity(4);
            auto x = np::ndarray<std::complex<double>>(std::vector<int>{4});
            for (int i = 0; i < 4; ++i)
              x[i] = {1, 0};
            auto y = mesh.apply(x);
            (void)y;
          }));
  printf(
      "neuromorphic encode: %.2f ms\n",
      ms(
          [&]
          {
            auto arr = np::ndarray<float>(std::vector<int>{64});
            for (int i = 0; i < 64; ++i)
              arr[i] = 0.5f;
            auto ea = np::spike::encode_rate(arr, 100, 100);
            (void)ea;
          }));
  printf(
      "padic Hensel: %.2f ms\n",
      ms(
          [&]
          {
            np::padic::Padic<int64_t> x0(7, 3, 10);
            auto f = [](const np::padic::Padic<int64_t>& x)
            { return np::padic::Padic<int64_t>(x.p, x.value * x.value - 2, x.prec); };
            auto df = [](const np::padic::Padic<int64_t>& x)
            { return np::padic::Padic<int64_t>(x.p, 2 * x.value, x.prec); };
            auto r = np::padic::HenselStrategy<int64_t>(5).lift(x0, f, df);
            (void)r;
          }));
  printf(
      "lattice LLL: %.2f ms\n",
      ms(
          [&]
          {
            auto lat = np::lattice::LatticeFactory::cubic<double>(4);
            auto r = lat.lll_reduce();
            (void)r;
          }));
  return 0;
}
