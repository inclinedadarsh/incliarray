#include "./include/NDArray.h"
#include <iostream>

int main() {
  NDArray a({2, 6});
  a.randint(1, 100);
  std::cout << "Matrix A:" << std::endl;
  a.print();

  // -------------------------

  NDArray b({1, 6});
  b.ones();
  b.set(4, 0);
  std::cout << std::endl << "Matrix B:" << std::endl;
  b.print();

  // -------------------------

  NDArray c = a + b;
  std::cout << std::endl << "Matrix C:" << std::endl;
  c.print();

  // -------------------------

  NDArray d = c.slice({{0, 2}, {2, 5}});
  std::cout << std::endl << "Matrix D:" << std::endl;
  d.print();

  NDArray e = a / b;
  std::cout << std::endl << "Matrix E:" << std::endl;
  e.print();

  NDArray f = a + 10;
  std::cout << std::endl << "Matrix F:" << std::endl;
  c.print();

  return 0;
}
