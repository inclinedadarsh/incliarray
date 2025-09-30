#include "./utils.h"
#include <stdexcept>

std::vector<int> detail::_computeStrides(std::vector<int> newShape) {
  std::vector<int> computedStrides;

  if (newShape.size() > 1) {
    for (int i = 0; i < newShape.size() - 1; i++) {
      int currentStrideValue = 1;
      for (int j = i + 1; j < newShape.size(); j++) {
        currentStrideValue *= newShape[j];
      }
      computedStrides.push_back(currentStrideValue);
    }
  }

  computedStrides.push_back(1);

  return computedStrides;
}

std::vector<int> detail::_broadcastShape(std::vector<int> a,
                                         std::vector<int> b) {
  std::vector<int> result;

  int aLength = a.size();
  int bLength = b.size();
  int resultNdim = std::max(aLength, bLength);

  for (int i = 0; i < resultNdim; i++) {
    int aDim = (i < resultNdim - aLength) ? 1 : a[i - (resultNdim - aLength)];
    int bDim = (i < resultNdim - bLength) ? 1 : b[i - (resultNdim - bLength)];

    if (aDim != bDim && aDim != 1 && bDim != 1) {
      throw std::invalid_argument("Shapes not broadcastable.");
    }

    result.push_back(std::max(aDim, bDim));
  }

  return result;
}

std::vector<int> detail::_broadcastStrides(std::vector<int> originalShape,
                                           std::vector<int> originalStrides,
                                           std::vector<int> targetShape) {
  int ndim = targetShape.size();
  int offset = ndim - originalShape.size();
  std::vector<int> result(ndim, 0);

  for (int i = 0; i < ndim; ++i) {
    if (i < offset) {
      result[i] = 0;
    } else if (originalShape[i - offset] == 1) {
      result[i] = 0;
    } else {
      result[i] = originalStrides[i - offset];
    }
  }

  return result;
}

int detail::_computeOffset(std::vector<int> index, std::vector<int> strides) {
  int offset = 0;
  for (size_t i = 0; i < index.size(); ++i) {
    offset += index[i] * strides[i];
  }
  return offset;
}
