#include "../include/NDArray.h"
#include <iostream>

int main(void) {
  NDArray X({1});
  X.randint(1, 10);

  std::cout << "X = ";
  X.print();

  NDArray Y = X * 2;
  NDArray Z = Y + 10;

  Z.backward();

  std::cout << "Grad X = ";
  X.print(NDArray::PrintType::Grad);

  return 0;
}
