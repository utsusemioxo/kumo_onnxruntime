#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "tensor.h"
#include "node.h"

namespace kumo {

struct Op {
  std::string type;
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

/**
 * @brief DAG 有向无环图
 * 
 */
class Graph {
public:
  bool LoadFromONNX(const std::string& onnx_path);
  void PrintGraph() const;
  void ExportToMermaid(std::ostream& out) const;
  void Run();
  std::vector<std::shared_ptr<Node>> TopoSort();
  void Forward();

  std::unordered_map<std::string, std::shared_ptr<Tensor>> tensor_map_;
  std::vector<std::shared_ptr<Node>> nodes_;
};
}