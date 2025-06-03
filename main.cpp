#include "incliarray.h"
#include <vector>

int main() {
  std::vector<int> shape = {5, 2, 3, 6};
  NDArray arr(shape);

  arr.metadata(true, true, true, true);

  return 0;
}
