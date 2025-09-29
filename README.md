
# incliarray

Minimal N-dimensional float array library in C++.  
Built from scratch to learn memory layout, strides, slicing, broadcasting, and autograd.

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white) ![Doxygen](https://img.shields.io/badge/doxygen-2C4AA8?style=for-the-badge&logo=doxygen&logoColor=white)

## Highlights

- Tiny, readable implementation (~single header+source) focused on concepts
- Explicit strides and offsets; safe guards around views/ownership
- Lightweight reverse-mode autograd for core ops

## Features

- N-dimensional float arrays with row‑major layout
- Manual stride-based indexing: `get`, `set`, flat `get(int)`, `set(int)`
- Introspection helpers: `metadata`, `isContiguous`, `print` (data or grads)
- Initialization/fill: `zeros`, `ones`, `fill`, `fillSequential`, `randint(low, high)`
- Slicing: `slice` returns a detached view (non-owning, no autograd) with shared data
- Reshape: `reshape` on contiguous, owning arrays
- Broadcasting arithmetic between arrays:
  - `+`, `-`, `/`, and element-wise `element_wise_multiply`
- Scalar arithmetic on arrays: `+ float`, `- float`, `/ float`, `element_wise_multiply(float)`
- Matrix multiplication for 2D arrays: `operator*` (no broadcasting)
- Autograd (reverse-mode) support:
  - Graph capture with `prev`, `op`, `label`
  - `backward()` builds a topo order and accumulates gradients
  - Implemented grads for: add/sub (array & scalar), div (array & scalar),
    element-wise multiply (array & scalar), and matrix multiply
- Safety and constraints:
  - Flat indexing and reshape allowed only on contiguous, owning arrays
  - Slice views do not own memory; fill operations disallowed on non-owning views
  - Division warns on divisor 0; grad contributions on zero divisors are skipped

## Build with CMake

1. Clone this repository and change directory
    ```bash
    git clone https://github.com/inclinedadarsh/incliarray.git
    cd incliarray
    ```

2. Create and change directory to the `build` directory
    ```bash
    mkdir build && cd build
    ```

3. Use `cmake` to build the program
    ```bash
    cmake ..
    make
    ```

4. Run the executable
    ```bash
    ./IncliArray
    ```

## Build Doxygen documentation

1. Install Doxygen and `graphviz`
    ```bash
    sudo apt install doxygen graphviz
    ```

2. Generate HTML docs
    ```bash
    doxygen Doxyfile
    ```

3. Open the docs in your browser. The index file can be found at `html/index.html`.

## Project Structure

```
.
├── CMakeLists.txt      // Build configuration
├── Doxyfile            // Doxygen config for documentation
├── include/            // Header files
│   ├── NDArray.h       // NDArray class declaration
│   └── utils.h         // Helper functions
├── src/                // Source files
│   ├── NDArray.cpp     // NDArray class implementation
│   └── utils.cpp       // Helper function definitions
├── main.cpp            // Example/test runner
├── LICENSE             // Project license
└── README.md           // Project documentation
```

## Quickstart Example

A compact example that demonstrates slicing, reshape, broadcasting, scalar ops, element‑wise ops, division, matrix multiplication, and a backward pass.

```cpp
#include "./include/NDArray.h"
#include <iostream>

int main() {
  NDArray A({2, 3});
  A.fillSequential();
  A.set({0, 1}, 10.0f);         // manual set using indices

  NDArray B({1, 3});
  B.ones();                      // broadcast row vector

  NDArray C = A + B;             // broadcasted add
  NDArray D = C.slice({{0, 2}, {1, 3}});  // view: shape (2, 2)
  
  NDArray E = D.element_wise_multiply(2.0f); // scalar mul
  NDArray F = E - 3.0f;            // scalar sub

  NDArray Bb = B.slice({{0, 1}, {1, 3}});    // shape (1, 2)
  NDArray G = F / (Bb + 1.0f);     // broadcasted div with scalar add

  NDArray W({2, 2});
  W.randint(1, 5);
  NDArray H = G * W;               // matrix multiply (2x2) * (2x2)

  H.metadata();
  std::cout << "H(1,1) = " << H.get({1, 1}) << std::endl;

  H.backward();                    // autograd

  std::cout << "\nGrad A:" << std::endl;
  A.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad B:" << std::endl;
  B.print(NDArray::PrintType::Grad);

  std::cout << "\nGrad W:" << std::endl;
  W.print(NDArray::PrintType::Grad);
}
```

## Author

Built with some neovim and some maths by [**Adarsh Dubey**](https://x.com/inclinedadarsh).

## License

MIT License. Do whatever you want, just don’t sue me.
