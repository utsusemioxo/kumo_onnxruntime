#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "onnx/onnx.pb.h"
#include "tensor.h"
#include <map>
#include <glog/logging.h>

namespace kumo {
struct Attribute {
  std::string name;
  enum class Type {INT, FLOAT, STRING, INTS, FLOATS, STRINGS, UNDEFINED} type;

  int64_t i;
  float f;
  std::string s;
  std::vector<int64_t> ints;
  std::vector<float> floats;
  std::vector<std::string> strings;
};

Attribute ConvertAttribute(const onnx::AttributeProto& onnx_attr);

} // namespace kumo