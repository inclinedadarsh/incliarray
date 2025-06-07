/**
 * @file utils.h
 * @brief Utility functions for shape and stride computations used in NDArray.
 *
 * This file contains helper functions within the `detail` namespace
 * for computing strides, broadcasting shapes and strides, and calculating
 * offsets.
 */
#pragma once

#include <vector>

namespace detail {
/**
 * @brief Computes strides from a new shape.
 * @param newShape Shape with which strides should be computed
 * @return Computed strides
 */
std::vector<int> _computeStrides(std::vector<int> newShape);

/**
 * @brief Computes the broadcasted shape of a and b together.
 * @param a First object for shape broadcasting
 * @param a Second object for shape broadcasting
 * @return Broadcasted shape
 */
std::vector<int> _broadcastShape(std::vector<int> a, std::vector<int> b);

/**
 * @brief Computes the broadcasted strides based on original shape and
 * strides, and target shape.
 * @param originalShape Original shape
 * @param originalStrides Original strides
 * @param targetShape The target shape. It is the shape which the new stride
 * will correspond to
 * @return Computed strides
 */
std::vector<int> _broadcastStrides(std::vector<int> originalShape,
                                   std::vector<int> originalStrides,
                                   std::vector<int> targetShape);

/**
 * @brief Computes offset based on the index and strides.
 * @param index Indices based on which the offset will be computed
 * @param strides Strides that will be used to compute the offset
 * @return Computed offset value
 */
int _computeOffset(std::vector<int> index, std::vector<int> strides);
} // namespace detail
