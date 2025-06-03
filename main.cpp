#include "incliarray.h"
#include <iostream>
#include <vector>

int main() {
  std::vector<int> shape = {5, 2, 3, 6};
  NDArray arr(shape);

  arr.metadata(true, true, true, true);

  std::vector<int> indices = {1, 1, 1, 1};
  arr.set(5, indices);
  std::cout << "Value at index [1, 1, 1, 1]: " << arr.get(indices) << std::endl;

  return 0;
}
