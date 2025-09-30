/**
 * @file NDArray.h
 * @brief NDArray: a minimal NumPy-like N-dimensional float array.
 *
 * This header declares the NDArray class, which provides:
 * - Row‑major storage with explicit shape and strides
 * - Safe element access via multi-index or flat index (when contiguous)
 * - Non-owning views via slice (detached from autograd)
 * - Reshape for owning, contiguous arrays
 * - Broadcasting arithmetic (+, -, /, element-wise multiply)
 * - Scalar arithmetic variants
 * - 2D matrix multiplication
 * - Lightweight reverse‑mode autograd for core ops
 *
 * Design notes:
 * - Memory ownership is explicit. Only base arrays (ownsData == true) allocate
 *   and manage their memory. Views never own.
 * - Autograd records lightweight operation metadata on results (except slices
 *   and clones, which are detached).
 * - Slices are detached views by default (no autograd participation). Use
 *   clone() to materialize an owning tensor when needed.
 */
#pragma once

#include <functional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

class NDArray {
private:
  /**
   * @brief Internal constructor used for creating non-owning views or
   *        wrapping existing memory with specific strides.
   *
   * This does not allocate memory. When ownsData == false, the lifetime of
   * the provided data pointer must outlive the NDArray instance.
   *
   * Autograd: callers decide whether to pass graph metadata. Slice uses this
   * constructor but returns a detached view (prev empty, op empty).
   */
  NDArray(std::vector<int> shape, std::vector<int> strides, float *data,
          bool ownsData, std::string label = "", std::string op = "",
          std::vector<std::reference_wrapper<NDArray>> prev = {});

  /**
   * @brief Build a topological ordering of nodes reachable from `arr`.
   *
   * DFS that collects unique nodes (by pointer) into `topo` such that parents
   * appear before children, enabling reverse‑mode backprop from outputs to
   * inputs.
   */
  void build_topo(std::unordered_set<NDArray *> &visited, NDArray *arr,
                  std::vector<std::reference_wrapper<NDArray>> &topo);

public:
  /** Raw data pointer in row‑major layout. Points to `size` floats. */
  float *data; /**< Raw data pointer in row‑major layout (length = size). */
  /** Dimensions of the array, e.g. {rows, cols} for 2D. */
  std::vector<int> shape; /**< Shape dimensions; product equals `size`. */
  /** Row‑major strides in elements; stride[i] is step for axis i. */
  std::vector<int> strides; /**< Row‑major strides per axis (in elements). */
  /** Number of dimensions (shape.size()). */
  int ndim; /**< Number of axes. */
  /** Total number of elements (product of `shape`). */
  int size = 0; /**< Total element count. */
  /** Whether this NDArray owns and manages the memory referenced by `data`.
   */
  bool ownsData; /**< True if this tensor allocated and owns its memory. */

  /** Gradient buffer aligned with logical indexing (allocated when
   * constructed). */
  float *grad; /**< Gradient storage parallel to `data`. */
  /** Operation tag for debug/inspection (e.g. "+", "-", "elem_mul", "*"). */
  std::string op; /**< Debug op tag. */
  /** Optional human‑readable label for this tensor. */
  std::string label; /**< User/debug label. */
  /** Parents in autograd graph. Empty for detached tensors (slice/clone). */
  std::vector<std::reference_wrapper<NDArray>> prev; /**< Graph parents. */
  /** Select between printing data or gradient buffers. */
  enum class PrintType { Data, Grad }; /**< Print selector. */
  /** Backpropagation closure; invoked during backward(). */
  std::function<void()> _backward; /**< Node‑local backward function. */

  /**
   * @brief Construct an owning, contiguous NDArray of given shape.
   *
   * Allocates `data` and `grad`, computes row‑major `strides`, and sets
   * `ownsData = true`. This constructor creates a base tensor that can
   * participate in autograd.
   */
  NDArray(std::vector<int> shape, std::string label = "", std::string op = "",
          std::vector<std::reference_wrapper<NDArray>> prev = {});

  /**
   * @brief Print selected metadata fields.
   * @param shapeInfo Whether to print shape
   * @param stridesInfo Whether to print strides (row‑major steps)
   * @param ndimInfo Whether to print number of dimensions
   * @param sizeInfo Whether to print total size (elements)
   * @param ownsDataInfo Whether to print ownership flag
   */
  void metadata(bool shapeInfo = true, bool stridesInfo = false,
                bool ndimInfo = false, bool sizeInfo = false,
                bool ownsDataInfo = false);

  /**
   * @brief Read an element by multi‑dimensional indices.
   * @param indices A vector of length `ndim` specifying the coordinate
   * @param type Whether to read from data or grad buffer
   * @return The element value
   * @throws std::invalid_argument if indices.size() != ndim
   */
  float get(std::vector<int> indices, PrintType type = PrintType::Data) const;

  /**
   * @brief Read an element by flat index.
   *
   * Valid only for contiguous, owning arrays. Treats memory as a flat buffer
   * regardless of shape, primarily for debugging and demos.
   *
   * @param index Flat position in [0, size)
   * @param type Whether to read data or grad
   * @return The element value
   * @throws std::out_of_range if index is outside [0, size)
   * @throws std::runtime_error if the array is not contiguous or not owning
   */
  float get(int index, PrintType type = PrintType::Data) const;

  /**
   * @brief Write an element by multi‑dimensional indices.
   * @param indices Coordinate of the element (length == ndim)
   * @param value Value to write into data buffer
   * @throws std::invalid_argument if indices.size() != ndim
   */
  void set(std::vector<int> indices, float value);

  /**
   * @brief Write an element by flat index.
   *
   * Valid only for contiguous, owning arrays. Writes into the underlying
   * memory ignoring shape.
   *
   * @param index Flat position in [0, size)
   * @param value Value to write
   * @throws std::out_of_range if index is outside [0, size)
   * @throws std::runtime_error if the array is not contiguous or not owning
   */
  void set(int index, float value);

  /**
   * @brief Return a non‑owning view restricted by per‑axis [start, stop)
   * slices.
   *
   * The returned NDArray shares `data` with the base tensor and has updated
   * shape/strides/offset. It is DETACHED from autograd: no graph is recorded
   * and its `_backward` is a no‑op.
   *
   * @param indices A vector (length == ndim) of (start, stop) pairs, inclusive
   *        start and exclusive stop for each axis
   * @return A detached view (non‑owning)
   * @throws std::invalid_argument if the number of slices != ndim
   */
  NDArray slice(std::vector<std::tuple<int, int>> indices);

  /**
   * @brief Whether the logical layout matches standard row‑major contiguous
   *        strides for the current shape.
   */
  bool isContiguous() const;

  /**
   * @brief Reshape this array to a new shape with the same number of elements.
   *
   * Only allowed for owning, contiguous arrays. Updates `shape` and recomputes
   * `strides` to standard row‑major.
   *
   * @param newShape Target shape (product must equal current size)
   * @throws std::runtime_error if the array is not owning or not contiguous
   * @throws std::invalid_argument if newShape is empty or incompatible in size
   */
  void reshape(std::vector<int> newShape);

  /**
   * @brief Pretty‑print the data or gradients.
   *
   * Prints 1D and 2D arrays in nested list form; higher‑dimensional arrays are
   * printed flattened for brevity.
   */
  void print(PrintType type = PrintType::Data);

  /**
   * @brief Fill with sequential values 0, 1, 2, ... (for demos/testing).
   * @throws std::runtime_error if the array does not own its memory
   */
  void fillSequential();

  /**
   * @brief Fill with a constant value.
   * @throws std::runtime_error if the array does not own its memory
   */
  void fill(float value);

  /**
   * @brief Set all elements to 0.
   */
  void zeros();

  /**
   * @brief Set all elements to 1.
   */
  void ones();

  /**
   * @brief Fill with uniform integer values in [low, high).
   * @param low Inclusive lower bound
   * @param high Exclusive upper bound
   * @throws std::runtime_error if the array does not own its memory
   */
  void randint(int low, int high);

  /**
   * @brief Broadcasted element‑wise addition (this + other).
   *
   * Autograd: result records graph metadata and accumulates dA += dOut and
   * dB += dOut with broadcasting.
   */
  NDArray operator+(NDArray &other);

  /**
   * @brief Scalar addition (this + value), shape‑preserving.
   */
  NDArray operator+(float value);

  /**
   * @brief Broadcasted element‑wise subtraction (this - other).
   *
   * Autograd: dA += dOut, dB += -dOut with broadcasting.
   */
  NDArray operator-(NDArray &other);

  /**
   * @brief Scalar subtraction (this - value), shape‑preserving.
   */
  NDArray operator-(float value);

  /**
   * @brief 2D matrix multiplication (no broadcasting).
   * @throws std::invalid_argument if either input is not 2D or dims mismatch
   * Autograd: implements dA = dC * B^T and dB = A^T * dC.
   */
  NDArray operator*(NDArray &other);

  /**
   * @brief Scalar element wise multiplication (no broadcasting).
   * @param value Value to multiply array with.
   */
  NDArray operator*(float value);

  /**
   * @brief Broadcasted element‑wise division (this / other).
   *
   * Warns on division by zero. Autograd: dA += dOut / other, dB += -(this /
   * other^2) * dOut.
   */
  NDArray operator/(NDArray &other);

  /**
   * @brief Scalar division (this / value), shape‑preserving, warns on zero.
   */
  NDArray operator/(float value);

  /**
   * @brief Scalar element-wise power (this ^ value).
   *
   * Raises each element to the given scalar power. Autograd: dA += value *
   * A^(value - 1) * dOut.
   */
  NDArray operator^(float value);

  /**
   * @brief Broadcasted element‑wise multiplication.
   *
   * Autograd: dA += other * dOut; dB += this * dOut.
   */
  NDArray element_wise_multiply(NDArray &other);

  /**
   * @brief Scalar element‑wise multiplication (this * value).
   */
  NDArray element_wise_multiply(float value);

  /**
   * @brief Reduce all elements to a scalar sum.
   *
   * Returns a 1-element NDArray holding the total sum. Autograd: distributes
   * the upstream gradient uniformly to every input element (dA += 1 * dOut).
   */
  NDArray sum();

  /**
   * @brief Sum along a specified axis (keep dimension as size 1).
   *
   * The output shape matches the input except `shape[axis] == 1`. Negative
   * axes are supported (Python-style indexing). Autograd: broadcasts upstream
   * gradient across the reduced axis into the input gradient.
   *
   * @param axis The axis along which to compute the sum (supports negatives)
   * @throws std::invalid_argument if axis is out of range after normalization
   */
  NDArray sum(int axis);

  /**
   * @brief Reverse‑mode backprop: accumulate gradients into all reachable
   *        parents from this node.
   *
   * Sets this->grad to ones (dOut/dOut = 1) and walks the graph in reverse
   * topological order, invoking each node’s `_backward` closure.
   */
  void backward();

  /**
   * @brief Materialize a contiguous, owning copy. Detached from autograd.
   *
   * If the source is contiguous, performs a fast contiguous copy; otherwise
   * uses a stride‑aware copy. The returned tensor has no graph linkage.
   */
  NDArray clone();
};
