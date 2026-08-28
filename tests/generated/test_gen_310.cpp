// Auto-generated test 310 - placeholder for future expansion
#include "np/np.hpp"
#include "test_util.hpp"
int main(){
  // Minimal sanity: ensure ndarray construction works
  auto a = np::ndarray<double>({2,2});
  a.fill(1.0);
  test::check(a.size()==4, "gen310 size");
  // dtype_t mapping sanity
  static_assert(std::is_same_v<np::dtype_t<np::float64>, double>);
  return test::failures()?1:0;
}
