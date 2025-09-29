#include "./include/NDArray.h"
#include <iostream>

int main() {
  // NDArray a({2, 6});
  // a.randint(1, 100);
  // std::cout << "Matrix A:" << std::endl;
  // a.print();
  //
  // // -------------------------
  //
  // NDArray b({1, 6});
  // b.ones();
  // b.set(4, 0);
  // std::cout << std::endl << "Matrix B:" << std::endl;
  // b.print();
  //
  // // -------------------------
  //
  // NDArray c = a + b;
  // std::cout << std::endl << "Matrix C:" << std::endl;
  // c.print();
  //
  // // -------------------------
  //
  // NDArray d = c.slice({{0, 2}, {2, 5}});
  // std::cout << std::endl << "Matrix D:" << std::endl;
  // d.print();
  //
  // NDArray e = a / b;
  // std::cout << std::endl << "Matrix E:" << std::endl;
  // e.print();
  //
  // NDArray f = a + 10;
  // std::cout << std::endl << "Matrix F:" << std::endl;
  // c.print();
  //
  // std::cout << "----------------------------" << std::endl;
  //
  // NDArray g({3, 4});
  // g.randint(1, 10);
  // std::cout << "Matrix G:" << std::endl;
  // g.print();
  //
  // NDArray h({4, 5});
  // h.randint(1, 10);
  // std::cout << "Matrix H:" << std::endl;
  // h.print();
  //
  // NDArray i = g * h;
  // std::cout << "Matrix I:" << std::endl;
  // i.print();

  NDArray a({2, 3});
  a.randint(1, 10);
  std::cout << "Matrix A:" << std::endl;
  a.print();

  NDArray b({3, 2});
  b.randint(1, 10);
  std::cout << "Matrix B:" << std::endl;
  b.print();

  NDArray c = a * b;
  std::cout << "Matrix C:" << std::endl;
  c.print();

  c.backward();

  std::cout << "Grad C:" << std::endl;
  c.print();

  std::cout << "\nGrad A:" << std::endl;
  a.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad B:" << std::endl;
  b.print(NDArray::PrintType::Grad);

  return 0;
}
