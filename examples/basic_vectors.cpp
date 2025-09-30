#include "../include/NDArray.h"
#include <iostream>

int main(void) {
  NDArray X({2, 3});
  X.fillSequential();

  std::cout << "X = " << std::endl;
  X.print();

  NDArray Y({3, 2});
  Y.randint(1, 10);

  std::cout << "\nY = " << std::endl;
  Y.print();

  NDArray Z = X * Y;

  std::cout << "\nZ = " << std::endl;
  Z.print();

  NDArray result = Z + 10;

  std::cout << "\nresult = " << std::endl;
  result.print();

  result.backward();

  std::cout << "\nGrad X = " << std::endl;
  X.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad Y = " << std::endl;
  Y.print(NDArray::PrintType::Grad);
}
