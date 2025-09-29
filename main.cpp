#include "./include/NDArray.h"
#include <iostream>

int main() {
  // Base tensor and initialization
  NDArray A({2, 3});
  A.fillSequential();
  A.set({0, 1}, 10.0f);
  std::cout << "A:" << std::endl;
  A.print();

  // Broadcasting row vector
  NDArray B({1, 3});
  B.ones();
  std::cout << "\nB:" << std::endl;
  B.print();

  // Broadcasted add then slice (view)
  NDArray C = A + B;                     // (2x3)
  NDArray D = C.slice({{0, 2}, {1, 3}}); // (2x2) view
  std::cout << "\nC = A + B:" << std::endl;
  C.print();
  std::cout << "\nD = C[:,1:3] (view):" << std::endl;
  D.print();

  // Scalar ops and element-wise multiply
  NDArray E = D.element_wise_multiply(2.0f);
  NDArray F = E - 3.0f;
  std::cout << "\nE = D * 2:" << std::endl;
  E.print();
  std::cout << "\nF = E - 3:" << std::endl;
  F.print();

  // Division with broadcasting (use a sub-slice of B)
  NDArray Bb = B.slice({{0, 1}, {1, 3}}); // (1x2)
  NDArray Tmp = Bb + 1.0f;
  NDArray G = F / Tmp;
  std::cout << "\nG = F / (Bb + 1):" << std::endl;
  G.print();

  // Matrix multiply
  NDArray W({2, 2});
  W.randint(1, 5);
  NDArray H = G * W; // (2x2) * (2x2)
  std::cout << "\nW:" << std::endl;
  W.print();
  std::cout << "\nH = G * W:" << std::endl;
  H.print();

  // Autograd
  H.backward();

  std::cout << "\nGrad A:" << std::endl;
  A.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad B:" << std::endl;
  B.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad W:" << std::endl;
  W.print(NDArray::PrintType::Grad);

  return 0;
}
