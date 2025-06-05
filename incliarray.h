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

  // To create an object (for general usage)
  NDArray(std::vector<int> shape);

  // To display the metadata information of the object
  void metadata(bool shapeInfo = true, bool stridesInfo = false,
                bool ndimInfo = false, bool sizeInfo = false,
                bool ownsDataInfo = false);

  // To get an element based on the indices
  float get(std::vector<int> indices);

  // To set an element based on the indices
  void set(float value, std::vector<int> indices);

  // To get a sliced object
  NDArray slice(std::vector<std::tuple<int, int>> indices);
};
