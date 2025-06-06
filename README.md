
# incliarray

Minimal NumPy-like N-dimensional float array library in C++.  
Built from scratch to understand memory layout, strides, slicing, and broadcasting.

## Examples

```cpp
NDArray A({2, 3});
A.fillSequential();   // A = [[0, 1, 2], [3, 4, 5]]

NDArray B({1, 3});
B.ones();             // B = [[1, 1, 1]]

NDArray C = A + B;    // Broadcasting + addition

C.metadata();         // View shape, strides, etc.
std::cout << C.get({1, 2}) << std::endl;  // prints 6

NDArray D = A.slice({{1, 2}, {0, 3}});
D.reshape({3});       // Reshape slice
D.fill(42);           // Fill with a single value
```

## Features

* N-dimensional float arrays
* Row-major memory layout
* Manual stride-based indexing
* Broadcasting with `+`, `-`, `*`
* Basic slicing (views, no copy)
* Reshape support
* Fill methods: `ones`, `zeroes`, `randint`, `fillSequential`
* Contiguity and memory ownership checks

## Build with CMake

```bash
git clone https://github.com/inclinedadarsh/incliarray.git
cd incliarray
mkdir build && cd build
cmake ..
make
./IncliArray
```

## Project Structure

```
incliarray/
├── CMakeLists.txt
├── incliarray.h      // Header with NDArray class
├── incliarray.cpp    // Implementation
└── main.cpp          // Example/test runner
```

## Roadmap

* [ ] `sum()` and `mean()` along axis
* [ ] Matrix multiplication
* [ ] Fancy indexing support
* [ ] Templated types (int, double, etc.)

## Author

Built with some neovim and some maths by [**Adarsh Dubey**](https://x.com/inclinedadarsh).

## 📜 License

MIT License. Do whatever you want, just don’t sue me.
