
# incliarray

Minimal NumPy-like N-dimensional float array library in C++.  
Built from scratch to understand memory layout, strides, slicing, and broadcasting.

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white) ![Doxygen](https://img.shields.io/badge/doxygen-2C4AA8?style=for-the-badge&logo=doxygen&logoColor=white)

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
incliarray/
├── CMakeLists.txt
├── incliarray.h      // Header with NDArray class
├── incliarray.cpp    // Implementation
└── main.cpp          // Example/test runner
```

## Author

Built with some neovim and some maths by [**Adarsh Dubey**](https://x.com/inclinedadarsh).

## License

MIT License. Do whatever you want, just don’t sue me.
