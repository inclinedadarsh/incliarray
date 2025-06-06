#include "incliarray.h"
#include <iostream>

int main() {
  NDArray a({2, 6});
  a.randint(10, 20);
  a.print();
  std::cout << "--------------------------" << std::endl;
  NDArray b({1, 6});
  b.ones();
  b.print();
  std::cout << "--------------------------" << std::endl;
  NDArray c = a + b;
  c.print();
  c.metadata(true, true, true, true, true);
  std::cout << "--------------------------" << std::endl;
  NDArray d = a - b;
  d.print();
  d.metadata(true, true, true, true, true);
  std::cout << "--------------------------" << std::endl;
  NDArray e = a * b;
  e.print();
  e.metadata(true, true, true, true, true);
  return 0;
}
