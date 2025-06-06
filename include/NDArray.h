/**
 * @file NDArray.h
 * @brief Defines the NDArray class for multidimensional array operations.
 *
 * NDArray supports element access, slicing, reshaping, broadcasting,
 * and arithmetic operations with efficient memory management.
 */
#pragma once

#include <tuple>
#include <vector>

class NDArray {
private:
  NDArray(std::vector<int> shape, std::vector<int> strides, float *data,
          bool ownsData);

public:
  float *data;
  std::vector<int> shape;
  std::vector<int> strides;
  int ndim;
  int size = 0;
  bool ownsData;

  NDArray(std::vector<int> shape);

  // ================
  // Public functions for general usage
  // ================

  /**
   * @brief Prints the metadata information of the object.
   * @param shapeInfo Print shape?
   * @param stridesInfo Print strides?
   * @param ndimInfo Print ndim?
   * @param sizeInfo Print size?
   * @param ownsDataInfo Print ownsData?
   */
  void metadata(bool shapeInfo = true, bool stridesInfo = false,
                bool ndimInfo = false, bool sizeInfo = false,
                bool ownsDataInfo = false);

  /**
   * @brief Returns an element based on the indices.
   * @param indices Indices of the element asked
   * @return Element at the given indices
   */
  float get(std::vector<int> indices);

  /**
   * @brief Returns an element based on a flat index.
   *
   * It disregards any shape or stride, and returns the element as if it was a
   * flat array.
   *
   * @param index Flat index of the element asked
   * @return Element at the given index
   */
  float get(int index);

  /**
   * @brief Sets a value of the element at the given indices.
   * @param indices Indices of the element where the value has to be set
   * @param value The value which is to be set at the given indices
   */
  void set(std::vector<int> indices, float value);

  /**
   * @brief Sets a value of the element at the given flat index.
   *
   * It disregards any shape or stride, and finds the element as if it was a
   * flat array.
   *
   * @param index Flat index of the element where the value has to be set
   * @param value The value which is to be set at the given flat index
   */
  void set(int index, float value);

  /**
   * @brief Returns a sliced object.
   *
   * Given the indices, this function returns an NDArray which is a view of the
   * original data but with the indices boundary.
   *
   * @param indices Vector (same size as the axis of the object) of tuple, where
   * the first element is start (inclusive) and second element is stop
   * (exclusive)
   *
   * @return Sliced object with new boundaries. This object DOES NOT own the
   * data
   */
  NDArray slice(std::vector<std::tuple<int, int>> indices);

  /**
   * @brief Returns boolean if the data inside is contiguous or not.
   * @return Boolean value, true if the data is contiguous, false otherwise
   */
  bool isContiguous();

  /**
   * @brief Reshapes the original data.
   *
   * The new shape should be compatible with the old shape.
   *
   * @param newShape The new shape in which the data has to be shaped
   */
  void reshape(std::vector<int> newShape);

  /**
   * @brief Prints the data.
   *
   * Prints the data in layout if it's 1d or 2d, otherwise prints the flattened
   * array.
   */
  void print();

  /**
   * @brief Fills the data in the object with sequential numbers starting from
   * 0.
   */
  void fillSequential();

  /**
   * @brief Fills the data inthe object with the given floating point value.
   * @param value The value to be used to fill the data
   */
  void fill(float value);

  /**
   * @brief Fills the data in the object with 0s.
   */
  void zeros();

  /**
   * @brief Fills the data in the object with 1s.
   */
  void ones();

  /**
   * @brief Fills the data in the object with random numbers between the given
   * boundaries.
   *
   * @param low The lower end of the boundary (inclusive)
   * @param high The higher end of the boundary (exclusive)
   */
  void randint(int low, int high);

  /**
   * @brief Performs broadcasted arithmetic addition operation.
   * @param other Other object taking part in addition operation
   * @return A new object with the result of the addition operation
   */
  NDArray operator+(const NDArray &other) const;

  /**
   * @brief Performs broadcasted arithmetic subtraction operation.
   * @param other Other object taking part in subtraction operation
   * @return A new object with the result of the subtraction operation
   */
  NDArray operator-(const NDArray &other) const;

  /**
   * @brief Performs broadcasted arithmetic multiplication operation.
   * @param other Other object taking part in multiplication operation
   * @return A new object with the result of the multiplication operation
   */
  NDArray operator*(const NDArray &other) const;
};
