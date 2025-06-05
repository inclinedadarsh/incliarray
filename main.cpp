#include "incliarray.h"
// #include <iostream>

int main() {
  NDArray a({2, 6});
  a.randint(10, 20);
  a.print();
  return 0;
}
