#pragma once
#include <string>
#include <vector>
#include <initializer_list>

namespace kumo {
struct Tensor {
  std::string name;
  std::vector<int> shape;
  std::vector<float> floatData;
  std::string dataType;

  Tensor() = default;
  Tensor(std::string name);
  Tensor(std::initializer_list<int> s);
  void fill(std::initializer_list<float> v);
  Tensor matmul(const Tensor& other) const;
  Tensor add(const Tensor& other) const;
  void print() const;
};
}