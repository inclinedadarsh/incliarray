#include "../include/NDArray.h"
#include "../include/utils.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

NDArray::NDArray(std::vector<int> inputShape, std::string inputLabel,
                 std::string inputOp,
                 std::vector<std::reference_wrapper<NDArray>> inputPrev) {
  // Initializing the shape
  shape = inputShape;

  // Initializing the size
  size = 1;

  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }

  // Initializing the data
  data = new float[size]();

  // Initializing the grad
  grad = new float[size]();

  // Initializing the strides
  strides = detail::_computeStrides(shape);

  // Initializing the ndim
  ndim = shape.size();

  // Initializing owns_data property
  ownsData = true;

  // Initializing label, op and prev
  label = inputLabel;
  op = inputOp;
  prev = inputPrev;

  // Default no-op backward
  _backward = []() {};
}

NDArray::NDArray(std::vector<int> inputShape, std::vector<int> inputStrides,
                 float *inputData, bool inputOwnsData, std::string inputLabel,
                 std::string inputOp,
                 std::vector<std::reference_wrapper<NDArray>> inputPrev) {
  shape = inputShape;
  strides = inputStrides;
  data = inputData;
  ownsData = inputOwnsData;
  label = inputLabel;
  op = inputOp;
  prev = inputPrev;

  size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }

  ndim = shape.size();

  // Initialize grad and default backward
  grad = new float[size]();
  _backward = []() {};
}

void NDArray::metadata(bool shapeInfo, bool stridesInfo, bool ndimInfo,
                       bool sizeInfo, bool ownsDataInfo) {
  // Printing the shape
  if (shapeInfo) {
    std::cout << "Shape of the array: (";
    for (int i = 0; i < shape.size() - 1; i++) {
      std::cout << shape[i] << ", ";
    }
    std::cout << shape[shape.size() - 1] << ")" << std::endl;
  }

  // Printing the strides
  if (stridesInfo) {
    std::cout << "Stride of the array: (";
    for (int i = 0; i < strides.size() - 1; i++) {
      std::cout << strides[i] << ", ";
    }
    std::cout << strides[strides.size() - 1] << ")" << std::endl;
  }

  // Printing the ndim
  if (ndimInfo) {
    std::cout << "NDim: " << ndim << std::endl;
  }

  // Printing the size
  if (sizeInfo) {
    std::cout << "Size: " << size << std::endl;
  }

  // Printing ownsData
  if (ownsDataInfo) {
    std::cout << "Owns Data: " << ownsData << std::endl;
  }
}

float NDArray::get(std::vector<int> indices, NDArray::PrintType type) const {
  // Check for size of input indices == ndim
  if (indices.size() != ndim) {
    throw std::invalid_argument("Expected " + std::to_string(ndim) +
                                " indices, got " +
                                std::to_string(indices.size()));
  }

  int offset = 0;
  for (int i = 0; i < ndim; i++) {
    offset += indices[i] * strides[i];
  }

  return type == NDArray::PrintType::Data ? data[offset] : grad[offset];
}

float NDArray::get(int index, NDArray::PrintType type) const {
  // Check for out of bound index
  if (index < 0 || index >= size) {
    throw std::out_of_range("Flat index out of bounds.");
  }

  // Only allow if the array is contiguous
  if (!isContiguous() || !ownsData) {
    throw std::runtime_error("Flat indexing only valid on base arrays.");
  }

  return type == NDArray::PrintType::Data ? data[index] : grad[index];
}

void NDArray::set(std::vector<int> indices, float value) {
  // Check for size of input indices == ndim
  if (indices.size() != ndim) {
    throw std::invalid_argument("Expected " + std::to_string(ndim) +
                                " indices, got " +
                                std::to_string(indices.size()));
  }

  int offset = 0;
  for (int i = 0; i < ndim; i++) {
    offset += indices[i] * strides[i];
  }

  data[offset] = value;
}

void NDArray::set(int index, float value) {
  // Check for out of bound index
  if (index < 0 || index >= size) {
    throw std::out_of_range("Flat index out of bounds.");
  }

  // Only allow if the array is contiguous
  if (!isContiguous() || !ownsData) {
    throw std::runtime_error("Flat indexing only valid on base arrays.");
  }

  data[index] = value;
}

NDArray NDArray::slice(std::vector<std::tuple<int, int>> slices) {
  if (slices.size() != ndim) {
    throw std::invalid_argument("Expected " + std::to_string(ndim) +
                                " slices, got " +
                                std::to_string(slices.size()));
  }

  int offset = 0;
  for (int i = 0; i < ndim; i++) {
    offset += std::get<0>(slices[i]) * strides[i];
  }

  std::vector<int> newShape;
  for (int i = 0; i < slices.size(); i++) {
    newShape.push_back(std::get<1>(slices[i]) - std::get<0>(slices[i]));
  }

  NDArray result(newShape, strides, data + offset, false);

  return result;
}

bool NDArray::isContiguous() const {
  std::vector<int> computedStrides = detail::_computeStrides(shape);

  if (computedStrides == strides)
    return true;
  else
    return false;
}

void NDArray::print(NDArray::PrintType type) {
  if (ndim == 1) {
    std::cout << "[";
    for (int i = 0; i < size - 1; i++) {
      if (type == NDArray::PrintType::Data) {
        std::cout << data[i] << ", ";
      } else {
        std::cout << grad[i] << ", ";
      }
    }
    if (type == NDArray::PrintType::Data)
      std::cout << data[size - 1] << "]" << std::endl;
    else
      std::cout << grad[size - 1] << "]" << std::endl;
  } else if (ndim == 2) {
    for (int i = 0; i < shape[0]; i++) {
      std::cout << "[";
      for (int j = 0; j < shape[1] - 1; j++) {
        std::cout << get({i, j}, type) << ", ";
      }
      std::cout << get({i, shape[1] - 1}, type) << "]" << std::endl;
    }
  } else {
    std::cout << "[";
    for (int i = 0; i < size - 1; i++) {
      if (type == NDArray::PrintType::Data)
        std::cout << data[i] << ", ";
      else
        std::cout << grad[i] << ", ";
    }
    if (type == NDArray::PrintType::Data)
      std::cout << data[size - 1] << "]" << std::endl;
    else
      std::cout << grad[size - 1] << "]" << std::endl;
  }
}

void NDArray::reshape(std::vector<int> newShape) {
  if (!isContiguous() || !ownsData) {
    throw std::runtime_error(
        "Reshaping is only allowed on contiguous and self-owned data.");
  }

  if (newShape.size() == 0) {
    throw std::invalid_argument(
        "The new shape should have al least one dimension, got 0.");
  }

  int newSize = 1;
  for (int i = 0; i < newShape.size(); i++) {
    newSize *= newShape[i];
  }

  if (newSize != size) {
    throw std::invalid_argument("New shape not compatible with the old shape.");
  }

  strides = detail::_computeStrides(newShape);
  shape = newShape;
}

void NDArray::fillSequential() {
  if (!ownsData) {
    throw std::runtime_error("Cannot fill a view or non-owning array.");
  }

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }
}

void NDArray::fill(float value) {
  if (!ownsData) {
    throw std::runtime_error("Cannot fill a view or non-owning array.");
  }

  for (int i = 0; i < size; i++) {
    data[i] = value;
  }
}

void NDArray::zeros() {
  if (!ownsData) {
    throw std::runtime_error("Cannot fill a view or non-owning array.");
  }
  fill(0.0);
}

void NDArray::ones() {
  if (!ownsData) {
    throw std::runtime_error("Cannot fill a view or non-owning array.");
  }
  fill(1.0);
}

void NDArray::randint(int low, int high) {
  if (!ownsData) {
    throw std::runtime_error("Cannot fill a view or non-owning array.");
  }

  std::mt19937 engine(std::random_device{}());
  std::uniform_int_distribution<int> dist(low, high - 1);

  for (int i = 0; i < size; i++) {
    data[i] = static_cast<float>(dist(engine));
  }
}

NDArray NDArray::operator+(NDArray &other) {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape, "", "+", {std::ref(*this), std::ref(other)});
  std::vector<int> index(outShape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offsetA = detail::_computeOffset(index, stridesA);
    int offsetB = detail::_computeOffset(index, stridesB);
    result.data[i] = this->data[offsetA] + other.data[offsetB];

    // Increment multi-dimensional index
    for (int dim = outShape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < outShape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward pass: dL/dA += 1 * dL/dOut (with broadcasting reduction)
  //                dL/dB += 1 * dL/dOut
  float *aGradPtr = this->grad;
  float *bGradPtr = other.grad;
  float *outGradPtr = result.grad;
  int outSize = result.size;
  std::vector<int> outShapeCopy = outShape;
  std::vector<int> stridesACopy = stridesA;
  std::vector<int> stridesBCopy = stridesB;

  result._backward = [aGradPtr, bGradPtr, outGradPtr, outSize, outShapeCopy,
                      stridesACopy, stridesBCopy]() mutable {
    std::vector<int> idx(outShapeCopy.size(), 0);
    for (int i = 0; i < outSize; ++i) {
      int offA = detail::_computeOffset(idx, stridesACopy);
      int offB = detail::_computeOffset(idx, stridesBCopy);
      float upstream = outGradPtr[i];
      aGradPtr[offA] += upstream;
      bGradPtr[offB] += upstream;

      for (int d = static_cast<int>(outShapeCopy.size()) - 1; d >= 0; --d) {
        idx[d]++;
        if (idx[d] < outShapeCopy[d])
          break;
        idx[d] = 0;
      }
    }
  };

  return result;
}

NDArray NDArray::operator+(float value) {
  NDArray result(shape, "", "+", {std::ref(*this)});
  std::vector<int> index(shape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offset = detail::_computeOffset(index, strides);
    result.data[i] = data[offset] + value;

    for (int dim = shape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < shape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward: dA += 1 * dOut
  float *aGradPtr = this->grad;
  float *outGradPtr = result.grad;
  int outSize = result.size;
  result._backward = [aGradPtr, outGradPtr, outSize]() mutable {
    for (int i = 0; i < outSize; ++i) {
      aGradPtr[i] += outGradPtr[i];
    }
  };

  return result;
}

NDArray NDArray::operator-(NDArray &other) {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape, "", "-", {std::ref(*this), std::ref(other)});
  std::vector<int> index(outShape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offsetA = detail::_computeOffset(index, stridesA);
    int offsetB = detail::_computeOffset(index, stridesB);
    result.data[i] = this->data[offsetA] - other.data[offsetB];

    for (int dim = outShape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < outShape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward pass: dL/dA += 1 * dL/dOut (with broadcasting reduction)
  //                dL/dB += -1 * dL/dOut
  float *aGradPtr = this->grad;
  float *bGradPtr = other.grad;
  float *outGradPtr = result.grad;
  int outSize = result.size;
  std::vector<int> outShapeCopy = outShape;
  std::vector<int> stridesACopy = stridesA;
  std::vector<int> stridesBCopy = stridesB;

  result._backward = [aGradPtr, bGradPtr, outGradPtr, outSize, outShapeCopy,
                      stridesACopy, stridesBCopy]() mutable {
    std::vector<int> idx(outShapeCopy.size(), 0);
    for (int i = 0; i < outSize; ++i) {
      int offA = detail::_computeOffset(idx, stridesACopy);
      int offB = detail::_computeOffset(idx, stridesBCopy);
      float upstream = outGradPtr[i];
      aGradPtr[offA] += upstream;
      bGradPtr[offB] -= upstream;

      for (int d = static_cast<int>(outShapeCopy.size()) - 1; d >= 0; --d) {
        idx[d]++;
        if (idx[d] < outShapeCopy[d])
          break;
        idx[d] = 0;
      }
    }
  };

  return result;
}

NDArray NDArray::operator-(float value) {
  NDArray result(shape, "", "-", {std::ref(*this)});
  std::vector<int> index(shape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offset = detail::_computeOffset(index, strides);
    result.data[i] = data[offset] - value;

    for (int dim = shape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < shape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward: dA += 1 * dOut (constant has no grad)
  float *aGradPtr = this->grad;
  float *outGradPtr = result.grad;
  int outSize = result.size;
  result._backward = [aGradPtr, outGradPtr, outSize]() mutable {
    for (int i = 0; i < outSize; ++i) {
      aGradPtr[i] += outGradPtr[i];
    }
  };

  return result;
}

NDArray NDArray::operator*(NDArray &other) {
  if (this->ndim != 2 || other.ndim != 2) {
    throw std::invalid_argument(
        "Matrix multiplication is only supported for 2d arrays! Exiting.");
  }

  if (this->shape[1] != other.shape[0]) {
    throw std::invalid_argument(
        "The column axis of first matrix and row axis of second matrix should "
        "be equal for matrix multiplication. Instead got " +
        std::to_string(this->shape[1]) +
        " for first matrix "
        "and " +
        std::to_string(other.shape[0]) + " for second matrix. Exiting.");
  }

  NDArray result({this->shape[0], other.shape[1]}, "", "*",
                 {std::ref(*this), std::ref(other)});

  for (int i = 0; i < result.shape[0]; i++) {
    for (int j = 0; j < result.shape[1]; j++) {
      float sum = 0.f;

      for (int k = 0; k < this->shape[1]; k++) {
        sum += get({i, k}) * other.get({k, j});
      }

      result.set({i, j}, sum);
    }
  }

  // Backward pass for matrix multiplication:
  // If C = A * B, then
  // dA += dC * B^T and dB += A^T * dC
  int m = this->shape[0];
  int k = this->shape[1];
  int n = other.shape[1];

  float *aGradPtr = this->grad;
  float *bGradPtr = other.grad;
  float *outGradPtr = result.grad;
  float *aDataPtr = this->data;
  float *bDataPtr = other.data;

  std::vector<int> aStridesCopy = this->strides;
  std::vector<int> bStridesCopy = other.strides;
  std::vector<int> outShapeCopy = {m, n};
  std::vector<int> outStridesCopy = detail::_computeStrides(outShapeCopy);

  result._backward = [m, k, n, aGradPtr, bGradPtr, outGradPtr, aDataPtr,
                      bDataPtr, aStridesCopy, bStridesCopy,
                      outStridesCopy]() mutable {
    // Accumulate into A's gradient: dA(i,k) += sum_j dC(i,j) * B(k,j)
    for (int i = 0; i < m; ++i) {
      for (int kk = 0; kk < k; ++kk) {
        float accum = 0.0f;
        for (int j = 0; j < n; ++j) {
          int offOut = detail::_computeOffset({i, j}, outStridesCopy);
          int offB = detail::_computeOffset({kk, j}, bStridesCopy);
          accum += outGradPtr[offOut] * bDataPtr[offB];
        }
        int offA = detail::_computeOffset({i, kk}, aStridesCopy);
        aGradPtr[offA] += accum;
      }
    }

    // Accumulate into B's gradient: dB(k,j) += sum_i A(i,k) * dC(i,j)
    for (int kk = 0; kk < k; ++kk) {
      for (int j = 0; j < n; ++j) {
        float accum = 0.0f;
        for (int i = 0; i < m; ++i) {
          int offOut = detail::_computeOffset({i, j}, outStridesCopy);
          int offAData = detail::_computeOffset({i, kk}, aStridesCopy);
          accum += aDataPtr[offAData] * outGradPtr[offOut];
        }
        int offB = detail::_computeOffset({kk, j}, bStridesCopy);
        bGradPtr[offB] += accum;
      }
    }
  };

  return result;
}

NDArray NDArray::operator/(NDArray &other) {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape, "", "/", {std::ref(*this), std::ref(other)});
  std::vector<int> index(outShape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offsetA = detail::_computeOffset(index, stridesA);
    int offsetB = detail::_computeOffset(index, stridesB);
    if (other.data[offsetB] == 0) {
      std::cerr << "\nWarning: Division by zero attempted. Result will be "
                   "'inf'."
                << std::endl;
    }
    result.data[i] = this->data[offsetA] / other.data[offsetB];

    for (int dim = outShape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < outShape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward pass: y = a / b =>
  // dA += (1/b) * dOut, dB += (-a / b^2) * dOut
  float *aGradPtr = this->grad;
  float *bGradPtr = other.grad;
  float *outGradPtr = result.grad;
  float *aDataPtr = this->data;
  float *bDataPtr = other.data;
  int outSize = result.size;
  std::vector<int> outShapeCopy = outShape;
  std::vector<int> stridesACopy = stridesA;
  std::vector<int> stridesBCopy = stridesB;

  result._backward = [aGradPtr, bGradPtr, outGradPtr, aDataPtr, bDataPtr,
                      outSize, outShapeCopy, stridesACopy,
                      stridesBCopy]() mutable {
    std::vector<int> idx(outShapeCopy.size(), 0);
    for (int i = 0; i < outSize; ++i) {
      int offA = detail::_computeOffset(idx, stridesACopy);
      int offB = detail::_computeOffset(idx, stridesBCopy);
      float upstream = outGradPtr[i];
      float aVal = aDataPtr[offA];
      float bVal = bDataPtr[offB];
      if (bVal != 0.0f) {
        aGradPtr[offA] += upstream / bVal;
        bGradPtr[offB] -= upstream * (aVal / (bVal * bVal));
      } else {
        // Follow same semantics as forward warning; avoid NaN in grad
        // accumulation By convention, treat contribution as 0 when dividing by
        // zero here
      }

      for (int d = static_cast<int>(outShapeCopy.size()) - 1; d >= 0; --d) {
        idx[d]++;
        if (idx[d] < outShapeCopy[d])
          break;
        idx[d] = 0;
      }
    }
  };

  return result;
}

NDArray NDArray::operator/(float value) {
  NDArray result(shape, "", "/", {std::ref(*this)});
  std::vector<int> index(shape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offset = detail::_computeOffset(index, strides);
    if (value == 0) {
      std::cerr << "\nWarning: Division by zero attempted. Result will be "
                   "'inf'."
                << std::endl;
    }
    result.data[i] = data[offset] / value;

    for (int dim = shape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < shape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward: y = a / c => dA += (1/c) * dOut
  float *aGradPtr = this->grad;
  float *outGradPtr = result.grad;
  int outSize = result.size;
  float c = value;
  result._backward = [aGradPtr, outGradPtr, outSize, c]() mutable {
    if (c == 0.0f)
      return; // already warned; skip accumulation to avoid NaNs
    for (int i = 0; i < outSize; ++i) {
      aGradPtr[i] += outGradPtr[i] / c;
    }
  };

  return result;
}

NDArray NDArray::element_wise_multiply(NDArray &other) {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape, "", "elem_mul", {std::ref(*this), std::ref(other)});
  std::vector<int> index(outShape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offsetA = detail::_computeOffset(index, stridesA);
    int offsetB = detail::_computeOffset(index, stridesB);
    result.data[i] = this->data[offsetA] * other.data[offsetB];

    for (int dim = outShape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < outShape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward pass: y = a * b => dA += b * dOut; dB += a * dOut
  float *aGradPtr = this->grad;
  float *bGradPtr = other.grad;
  float *outGradPtr = result.grad;
  float *aDataPtr = this->data;
  float *bDataPtr = other.data;
  int outSize = result.size;
  std::vector<int> outShapeCopy = outShape;
  std::vector<int> stridesACopy = stridesA;
  std::vector<int> stridesBCopy = stridesB;

  result._backward = [aGradPtr, bGradPtr, outGradPtr, aDataPtr, bDataPtr,
                      outSize, outShapeCopy, stridesACopy,
                      stridesBCopy]() mutable {
    std::vector<int> idx(outShapeCopy.size(), 0);
    for (int i = 0; i < outSize; ++i) {
      int offA = detail::_computeOffset(idx, stridesACopy);
      int offB = detail::_computeOffset(idx, stridesBCopy);
      float upstream = outGradPtr[i];
      aGradPtr[offA] += upstream * bDataPtr[offB];
      bGradPtr[offB] += upstream * aDataPtr[offA];

      for (int d = static_cast<int>(outShapeCopy.size()) - 1; d >= 0; --d) {
        idx[d]++;
        if (idx[d] < outShapeCopy[d])
          break;
        idx[d] = 0;
      }
    }
  };

  return result;
}

NDArray NDArray::element_wise_multiply(float value) {
  NDArray result(shape, "", "elem_mul", {std::ref(*this)});
  std::vector<int> index(shape.size(), 0);

  for (int i = 0; i < result.size; ++i) {
    int offset = detail::_computeOffset(index, strides);
    result.data[i] = data[offset] * value;

    for (int dim = shape.size() - 1; dim >= 0; --dim) {
      index[dim]++;
      if (index[dim] < shape[dim])
        break;
      index[dim] = 0;
    }
  }

  // Backward: y = a * c => dA += c * dOut
  float *aGradPtr = this->grad;
  float *outGradPtr = result.grad;
  int outSize = result.size;
  float c = value;
  result._backward = [aGradPtr, outGradPtr, outSize, c]() mutable {
    for (int i = 0; i < outSize; ++i) {
      aGradPtr[i] += outGradPtr[i] * c;
    }
  };

  return result;
}

void NDArray::build_topo(std::unordered_set<NDArray *> &visited, NDArray *arr,
                         std::vector<std::reference_wrapper<NDArray>> &topo) {
  if (visited.find(arr) == visited.end()) {
    visited.insert(arr);
    for (auto &p : arr->prev) {
      build_topo(visited, &p.get(), topo);
    }
    topo.push_back(*arr);
  }
}

void NDArray::backward() {
  std::unordered_set<NDArray *> visited;
  std::vector<std::reference_wrapper<NDArray>> topo;
  build_topo(visited, this, topo);

  // Initialize gradient of the output w.r.t itself to ones
  for (int i = 0; i < size; ++i) {
    grad[i] = 1.0f;
  }

  for (int i = static_cast<int>(topo.size()) - 1; i >= 0; --i) {
    topo[i].get()._backward();
  }
}
