#include "../include/NDArray.h"
#include "../include/utils.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

NDArray::NDArray(std::vector<int> inputShape) {
  // Initializing the shape
  shape = inputShape;

  // Initializing the size
  size = 1;

  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }

  // Initializing the data
  data = new float[size]();

  // Initializing the strides
  strides = detail::_computeStrides(shape);

  // Initializing the ndim
  ndim = shape.size();

  // Initializing owns_data property
  ownsData = true;
}

NDArray::NDArray(std::vector<int> inputShape, std::vector<int> inputStrides,
                 float *inputData, bool inputOwnsData) {
  shape = inputShape;
  strides = inputStrides;
  data = inputData;
  ownsData = inputOwnsData;

  size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }

  ndim = shape.size();
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

float NDArray::get(std::vector<int> indices) const {
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

  return data[offset];
}

float NDArray::get(int index) const {
  // Check for out of bound index
  if (index < 0 || index >= size) {
    throw std::out_of_range("Flat index out of bounds.");
  }

  // Only allow if the array is contiguous
  if (!isContiguous() || !ownsData) {
    throw std::runtime_error("Flat indexing only valid on base arrays.");
  }

  return data[index];
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

void NDArray::print() {
  if (ndim == 1) {
    std::cout << "[";
    for (int i = 0; i < size - 1; i++) {
      std::cout << data[i] << ", ";
    }
    std::cout << data[size - 1] << "]" << std::endl;
  } else if (ndim == 2) {
    for (int i = 0; i < shape[0]; i++) {
      std::cout << "[";
      for (int j = 0; j < shape[1] - 1; j++) {
        std::cout << get({i, j}) << ", ";
      }
      std::cout << get({i, shape[1] - 1}) << "]" << std::endl;
    }
  } else {
    std::cout << "[";
    for (int i = 0; i < size - 1; i++) {
      std::cout << data[i] << ", ";
    }
    std::cout << data[size - 1] << "]" << std::endl;
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

NDArray NDArray::operator+(const NDArray &other) const {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape);
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

  return result;
}

NDArray NDArray::operator+(float value) const {
  NDArray result(shape);
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

  return result;
}

NDArray NDArray::operator-(const NDArray &other) const {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape);
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

  return result;
}

NDArray NDArray::operator-(float value) const {
  NDArray result(shape);
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

  return result;
}

NDArray NDArray::operator/(const NDArray &other) const {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape);
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

  return result;
}

NDArray NDArray::operator/(float value) const {
  NDArray result(shape);
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

  return result;
}

NDArray NDArray::element_wise_multiply(const NDArray &other) const {
  std::vector<int> outShape = detail::_broadcastShape(shape, other.shape);
  std::vector<int> stridesA =
      detail::_broadcastStrides(shape, strides, outShape);
  std::vector<int> stridesB =
      detail::_broadcastStrides(other.shape, other.strides, outShape);

  NDArray result(outShape);
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

  return result;
}

NDArray NDArray::element_wise_multiply(float value) const {
  NDArray result(shape);
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

  return result;
}
