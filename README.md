# IncliArray

Minimal N-dimensional float array library in C++.

Checkout the documentation: https://inclinedadarsh.github.io/incliarray/.

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white) ![Doxygen](https://img.shields.io/badge/doxygen-2C4AA8?style=for-the-badge&logo=doxygen&logoColor=white)

## Highlights

- Tiny, readable implementation (single header + single source) focused on concepts
- Explicit shapes/strides and careful ownership rules (base vs view)
- Lightweight reverse‑mode autograd for core ops

## Features

- N‑dimensional float arrays with row‑major layout
- Safe, stride‑aware indexing:
  - `get(std::vector<int>)`, `set(std::vector<int>, float)`
  - Flat `get(int)`, `set(int)` on contiguous, owning arrays only
- Introspection and layout helpers: `metadata`, `isContiguous`, `print` (data or grads)
- Initialization/fill:
  - `zeros`, `ones`, `fill`, `fillSequential`
  - `randint(low, high)`, `rand()` in [0,1), `rand(low, high)`
- Views and materialization:
  - `slice` returns a detached, non‑owning view (shares data, no autograd linkage)
  - `clone()` creates a contiguous, owning copy (detached)
- Reshape: `reshape(newShape)` for contiguous, owning arrays
- Broadcasting arithmetic (array ⊕ array): `+`, `-`, `/`, `element_wise_multiply`
- Scalar arithmetic (array ⊕ scalar): `+ float`, `- float`, `/ float`, `element_wise_multiply(float)`
- Element‑wise power with scalar exponent: `operator^(float)`
- 2D matrix multiplication: `operator*` (no broadcasting)
- Reductions:
  - `sum()` reduces all elements to a 1‑element array
  - `sum(axis)` keeps reduced dimension as size 1, supports negative axes
- Autograd:
  - Results from core ops capture `prev`, `op`, `label`
  - `backward()` builds a topological order and accumulates gradients
  - Implemented grads for add/sub (array & scalar), div (array & scalar),
    element‑wise multiply (array & scalar), matrix multiply, and power (scalar exponent)
- Safety/constraints:
  - Flat indexing and reshape are allowed only on contiguous, owning arrays
  - Views are non‑owning; fill operations are disallowed on non‑owning arrays
  - Division warns on divisor 0; gradient contributions on zero divisors are skipped

## Build and run examples (development flow)

```bash
mkdir build && cd build
cmake ..
make
./examples/basic_scalars
```

Examples are built from `examples/` when `BUILD_EXAMPLES` is ON (default). You can also run `./examples/basic_vectors` if present.

See `examples/README.md` for details about each example.

## Install the library system‑wide

```bash
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install
```

This installs headers to `/usr/local/include` and the library to `/usr/local/lib` by default.

## Use in your project

### With CMake

In your project's `CMakeLists.txt`:

```cmake
find_package(NDArray REQUIRED)
add_executable(myapp myapp.cpp)
target_link_libraries(myapp PRIVATE NDArray)
```

Then include it in code as:

```cpp
#include <NDArray.h>
```

### With a plain compiler (no CMake)

If installed to default prefixes, this typically works:

```bash
g++ -std=c++17 myapp.cpp -lNDArray -o myapp
```

And in your code:

```cpp
#include <NDArray.h>
```

If you installed to a custom prefix, add `-I<include_dir>` and `-L<lib_dir>` accordingly.

## Doxygen documentation

1) Install Doxygen and `graphviz`:

```bash
sudo apt-get install -y doxygen graphviz
```

2) Generate docs:

```bash
./build-docs.sh
```

Open `html/index.html` in your browser.

## Project structure

```
.
├── CMakeLists.txt       // Library build + install rules
├── Doxyfile             // Doxygen configuration
├── include/
│   ├── NDArray.h        // NDArray class declaration
│   └── utils.h          // Internal helpers (strides, offsets, broadcasting)
├── src/
│   ├── NDArray.cpp      // NDArray implementation
│   └── utils.cpp        // Helper implementations
├── examples/
│   ├── CMakeLists.txt   // Example build targets
│   ├── basic_scalars.cpp
│   └── basic_vectors.cpp
├── LICENSE
└── README.md
```

## Quickstart demo

```cpp
#include <NDArray.h>
#include <iostream>

int main() {
  NDArray A({2, 3});
  A.fillSequential();

  NDArray B({1, 3});
  B.ones();

  NDArray C = A + B;                       // broadcast add
  NDArray D = C.slice({{0, 2}, {1, 3}});   // detached view (2x2)
  NDArray E = D.element_wise_multiply(2.0f);
  NDArray F = E - 3.0f;

  NDArray W({2, 2});
  W.randint(1, 5);
  NDArray H = F * W;                        // 2D matmul

  H.backward();
  A.print(NDArray::PrintType::Grad);
}
```

## Author

Built with some neovim and some maths by [**Adarsh Dubey**](https://x.com/inclinedadarsh).

## License

MIT License. Do whatever you want, just don’t sue me.
