/**
 * @file python/numpy_cpp.cpp
 * @brief pybind11 bridge — np::ndarray ↔ numpy.ndarray zero-copy, linalg, lattice, padic, hardware.
 *
 * Build: cmake -DNP_BUILD_PYTHON=ON -S . -B build && cmake --build build
 * Via pip: pip install ./python
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <np/np.hpp>

namespace py = pybind11;
using namespace np;

PYBIND11_MODULE(numpy_cpp, m)
{
  m.doc() = "numpy-cpp Python bridge — header-only C++20 NumPy 2.2 via pybind11";

  m.def("arange", [](double start, double stop, double step) { return arange<double>(start, stop, step); }, "arange");
  m.def("zeros", [](std::vector<int> shape) { return zeros<double>(shape); }, "zeros");
  m.def("ones", [](std::vector<int> shape) { return ones<double>(shape); }, "ones");
  m.def("eye", [](int n) { return eye<double>(n); }, "eye");

  // linalg
  auto mlinalg = m.def_submodule("linalg", "np::linalg");
  mlinalg.def("matmul", [](const ndarray<double>& a, const ndarray<double>& b) { return linalg::matmul(a, b); });
  mlinalg.def("norm", [](const ndarray<double>& a) { return linalg::norm(a); });

  // lattice
  auto mlattice = m.def_submodule("lattice", "np::lattice");
  mlattice.def("cubic", [](int n) { return lattice::LatticeFactory::cubic<double>(n); });
  mlattice.def("lll", [](const lattice::Lattice<double>& lat) { return lat.lll_reduce(); });

  // padic
  auto mpadic = m.def_submodule("padic", "np::padic");
  mpadic.def("padic", [](int p, int64_t v, int prec) { return padic::Padic<int64_t>(p, v, prec); });
  mpadic.def("valuation", [](const padic::Padic<int64_t>& a) { return a.valuation(); });

  // hardware
  auto mhw = m.def_submodule("hardware", "accelerator/neuromorphic/tensor/mem");
  mhw.def("hbm_migrate", [](const ndarray<float>& a) { return mem::migrate_to_hbm(a).data; });
  auto mneuro = mhw.def_submodule("neuromorphic", "Loihi/SpiNNaker");
  mneuro.def("encode_rate", [](const ndarray<float>& a) { return spike::encode_rate(a); });
  auto mtensor = mhw.def_submodule("tensor", "Hopper/AMX");
  mtensor.def("matmul_fp8", [](const ndarray<float>& a, const ndarray<float>& b) { return tensor::matmul_fp8(a, b); });
  auto mquantum = mhw.def_submodule("quantum", "StateVector");
  mquantum.def("plus_state", [](int n) { return quantum::QuantumFactory::plus_state(n); });
}
