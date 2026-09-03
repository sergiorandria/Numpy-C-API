/**
 * @example physics_navier_stokes.cpp
 * Navier-Stokes 2D lid-driven cavity via np::physics
 */
#include <np/np.hpp>
#include <iostream>

int main()
{
  auto ns = np::physics::PhysicsFactory::navier_stokes(32, 32, 100);
  ns.state.u(16, 16) = 1.0;
  for (int i = 0; i < 5; ++i)
    ns.step();
  std::cout << "ke " << ns.kinetic_energy() << " div " << ns.max_divergence() << "\n";
  auto builder = np::physics::SolverBuilder::create().size(16, 16).reynolds(200).build();
  std::cout << "builder " << builder.state.nx << "x" << builder.state.ny << "\n";
  return 0;
}
