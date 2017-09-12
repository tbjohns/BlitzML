#include <blitzml/base/math_util.h>

#include <utility>

using std::pair;

namespace BlitzML {


pair<value_t, value_t> compute_quadratic_roots(value_t a, value_t b, 
                                                    value_t c) {
  value_t discriminant = b*b - 4*a*c;
  bool result_exists = (discriminant >= 0);
  if (a == 0. && b == 0. && c != 0.) {
    result_exists = false;
  }
  if (!result_exists) {
    value_t nan = std::numeric_limits<value_t>::quiet_NaN();
    return pair<value_t, value_t>(nan, nan);
  } 
  if (a == 0.) {
    value_t result = -c/b;
    return pair<value_t, value_t>(result, result);
  }

  value_t sqrt_discriminant = sqrt(discriminant);
  value_t root1 = (-b - sqrt_discriminant) / (2 * a);
  value_t root2 = (-b + sqrt_discriminant) / (2 * a);
  if (a > 0) {
    return pair<value_t, value_t>(root1, root2);
  } else {
    return pair<value_t, value_t>(root2, root1);
  }
}

} // namespace BlitzML

