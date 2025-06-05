#pragma once

#include <tuple>
#include <vector>

class NDArray {
private:
  NDArray(std::vector<int> shape, std::vector<int> strides, float *data,
          bool ownsData);

  /*
    Private utility functions
    These functions start with an underscore.
  */

  // To compute strides based on the current shape
  std::vector<int> _compute_strides();
  // Compute strides for a new shape
  static std::vector<int> _compute_strides(std::vector<int> newShape);

public:
  float *data;
  std::vector<int> shape;
  std::vector<int> strides;
  int ndim;
  int size = 0;
  bool ownsData;

  /*
   Public functions for usage
  */

  // To create an object (for general usage)
  NDArray(std::vector<int> shape);

  // To display the metadata information of the object
  void metadata(bool shapeInfo = true, bool stridesInfo = false,
                bool ndimInfo = false, bool sizeInfo = false,
                bool ownsDataInfo = false);

  // To get an element based on the indices
  float get(std::vector<int> indices);

  // To get an element based on a flat index
  float get(int index);

  // To set an element based on the indices
  void set(std::vector<int> indices, float value);

  // To set an element based on a flat index
  void set(int index, float value);

  // To get a sliced object
  NDArray slice(std::vector<std::tuple<int, int>> indices);

  // Returns if if the data is contiguous or not
  bool isContiguous();

  // Reshapes the original array
  void reshape(std::vector<int> newShape);

  // Print the array
  void print();

  // Fill the array with sequential data
  void fillSequential();

  // Fill the data with a constant value
  void fill(float value);

  // Fill the data with 0s
  void zeros();

  // Fill the data with 1s
  void ones();

  // Fill the data with random numbers
  void randint(int low, int high);
};
