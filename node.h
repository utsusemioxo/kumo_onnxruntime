#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensor.h"
#include <map>
#include "attribute.h"
#include <glog/logging.h>

namespace kumo {
struct Node {
public:
  std::string name;
  std::string op_type;
  std::vector<std::string> inputs; // input tensor name
  std::vector<std::string> outputs; // output tensor name
  std::map<std::string, Attribute> attributes;
};

} // namespace kumo