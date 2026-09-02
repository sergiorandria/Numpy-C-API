/**
 * @file test_accelerator.cpp
 */
#include <np/np.hpp>
#include "test_util.hpp"
int main()
{
  using namespace np::accelerator;
  auto cpu = AcceleratorFactory::cpu();
  auto gpu = AcceleratorFactory::gpu();
  auto loihi = AcceleratorFactory::loihi();
  auto reram = AcceleratorFactory::reram();
  test::check(cpu->name() == "CPU", "CPU");
  test::check(gpu->name() == "GPU", "GPU");
  test::check(loihi->name() == "Loihi2", "Loihi");
  test::check(reram->name() == "ReRAM", "ReRAM");
  auto a = np::eye<float>(2);
  auto b = np::eye<float>(2);
  auto c = cpu->matmul(a, b);
  test::check(c.size() == 4, "accelerator matmul");
  return test::failures() ? 1 : 0;
}
