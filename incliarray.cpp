#include "incliarray.h"
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

/*
 Private utility functions
 These function start with an underscore.
*/

std::vector<int> NDArray::_compute_strides() {
  std::vector<int> computedStrides;

  if (shape.size() > 1) {
    for (int i = 0; i < shape.size() - 1; i++) {
      int currentStrideValue = 1;
      for (int j = i + 1; j < shape.size(); j++) {
        currentStrideValue *= shape[j];
      }
      computedStrides.push_back(currentStrideValue);
    }
  }

  computedStrides.push_back(1);

  return computedStrides;
}

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
  strides = _compute_strides();

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

float NDArray::get(std::vector<int> indices) {
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

float NDArray::get(int index) {
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

bool NDArray::isContiguous() {
  std::vector<int> computedStrides = _compute_strides();

  if (computedStrides == strides)
    return true;
  else
    return false;
}
