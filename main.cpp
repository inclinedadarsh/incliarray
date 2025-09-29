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

  // Broadcasted add
  NDArray C = A + B; // (2x3)
  std::cout << "\nC = A + B:" << std::endl;
  C.print();

  // Scalar ops and element-wise multiply
  NDArray E = C.element_wise_multiply(2.0f);
  NDArray F = E - 3.0f;
  std::cout << "\nE = D * 2:" << std::endl;
  E.print();
  std::cout << "\nF = E - 3:" << std::endl;
  F.print();

  // Division with broadcasting
  NDArray Tmp = B + 1.0f; // (1x3)
  NDArray G = F / Tmp;    // (2x3) / (1x3)
  std::cout << "\nG = F / (B + 1):" << std::endl;
  G.print();

  // Matrix multiply
  NDArray W({3, 2});
  W.randint(1, 5);
  NDArray H = G * W; // (2x3) * (3x2) -> (2x2)
  std::cout << "\nW:" << std::endl;
  W.print();
  std::cout << "\nH = G * W:" << std::endl;
  H.print();

  // Autograd
  H.backward();

  std::cout << "\nGrad H:" << std::endl;
  H.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad A:" << std::endl;
  A.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad B:" << std::endl;
  B.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad C:" << std::endl;
  C.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad E:" << std::endl;
  E.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad F:" << std::endl;
  F.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad Tmp:" << std::endl;
  Tmp.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad G:" << std::endl;
  G.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad W:" << std::endl;
  W.print(NDArray::PrintType::Grad);

  return 0;
}
