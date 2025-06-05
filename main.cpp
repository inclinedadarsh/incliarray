#include "incliarray.h"
#include <iostream>

int main() {
  NDArray arr({5, 6});

  arr.metadata(true, true, true, true, true);

  arr.set({1, 1}, 5);
  std::cout << "Value at index [1, 1]: " << arr.get({1, 1}) << std::endl;

  std::cout << "-----------------------" << std::endl;

  NDArray newArr = arr.slice({{2, 4}, {2, 5}});
  newArr.metadata(true, true, true, true, true);

  std::cout << "-----------------------" << std::endl;

  arr.print();

  return 0;
}
