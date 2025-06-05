#include "tensor.h"
#include <initializer_list>
#include <iostream>

namespace kumo {
Tensor::Tensor(std::string name) {
  
}
Tensor::Tensor(std::initializer_list<int> s) : shape(s), data() {
  int size = 1;
  for (int dim : s) size *= dim;
  data.resize(size);
}

void Tensor::fill(std::initializer_list<float> v) {
  data.assign(v.begin(), v.end());
}

Tensor Tensor::matmul(const Tensor& other) const {
  // TODO: Implement matmul
  return *this;
}

Tensor Tensor::add(const Tensor& other) const {
  // TODO: Implement add
  return *this;
}

void Tensor::print() const {
  for (float val : data) std::cout << val << " ";
  std::cout << std::endl;
}

}