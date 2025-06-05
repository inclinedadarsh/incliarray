#include "incliarray.h"
#include <iostream>

int main() {
  NDArray a({2, 6});
  a.set({1, 0}, 1);

  a.metadata();
  a.print();

  a.reshape({3, 4});
  a.print();
  std::cout << a.get({1, 2});
  std::cout << a.get(6);
  return 0;
}
