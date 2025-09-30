# Examples

This directory contains small programs demonstrating `NDArray` usage.

## Build

From the project root:

```bash
mkdir build && cd build
cmake ..
make
```

Executables will appear under `./examples/` in the build directory.

## Run

### basic_scalars

Demonstrates scalar arithmetic and autograd on a 1‑element array:

```bash
./examples/basic_scalars
```

What it does:
- Creates a shape `{1}` array, random integer fill
- Applies `* 2`, then `+ 10`
- Calls `backward()` and prints gradient of the input

### basic_vectors

Demonstrates matrix multiplication, broadcasting with scalars, and autograd:

```bash
./examples/basic_vectors
```

What it does:
- Creates `X` with shape `{2,3}` and sequential values
- Creates `Y` with shape `{3,2}` and random integers
- Computes `Z = X * Y` (2D matmul)
- Computes `result = Z + 10` (scalar broadcast)
- Runs `backward()` and prints gradients for `X` and `Y`


