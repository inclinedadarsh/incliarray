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
  void metadata(bool shapeInfo = true, bool stridesInfo = false,
                bool ndimInfo = false, bool sizeInfo = false,
                bool ownsDataInfo = false);

  float get(std::vector<int> indices);
  void set(float value, std::vector<int> indices);
  NDArray slice(std::vector<std::tuple<int, int>> indices);
};
