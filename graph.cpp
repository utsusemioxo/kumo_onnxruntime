#include "graph.h"
#include "attribute.h"
#include "node.h"
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <onnx/onnx.pb.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

namespace kumo {
bool Graph::LoadFromONNX(const std::string& onnx_path) {
  onnx::ModelProto model;
  std::ifstream input_onnx(onnx_path, std::ios::binary);
  if (!input_onnx) {
    LOG(FATAL) << "Failed to open file!";
  }
  model.ParseFromIstream(&input_onnx);

  // convert nodes_
  const auto& graph_proto = model.graph();
  for (const auto& onnx_node : graph_proto.node()) {
    auto node = std::make_shared<Node>();
    node->name = onnx_node.name();
    node->op_type = onnx_node.op_type();

    for (const auto& input : onnx_node.input()) {
      node->inputs.push_back(input);
    }

    for (const auto& output : onnx_node.output()) {
      node->outputs.push_back(output);
    }

    for (const auto& attr : onnx_node.attribute()) {
      node->attributes[attr.name()] = ConvertAttribute(attr);
    }

    nodes_.push_back(std::move(node));
  }
  
  return true;
}

void Graph::PrintGraph() const {
  LOG(INFO) << "Printing graph structure...";
  for (const auto& node : nodes_) {
    LOG(INFO) << "Node: " << node->name << " [" << node->op_type << "]";
    for (const auto& in : node->inputs) {
      LOG(INFO) << "    input: " << in;
    }
    for (const auto& out : node->outputs) {
      LOG(INFO) << "    output: " << out;
    }
  }
}

void Graph::ExportToMermaid(std::ostream& out) const{
  out << "graph TD\n";

  // 标记所有节点
  for (const auto& node : nodes_) {
    std::string node_id = "node_" + node->name;
    out << "  " << node_id << "[\"" << node->op_type << "\\n" << node->name << "\"]\n";
  }

  // 连接边：input -> node
  for (const auto& node : nodes_) {
    std::string node_id = "node_" + node->name;
    for (const auto& input : node->inputs) {
      out << "  " << input << " --> " << node_id << "\n";
    }
    for (const auto& output : node->outputs) {
      out << "  " << node_id << " --> " << output << "\n";
    }
  }



  // 添加样式（可选）
  // out << "  classDef input fill:#e0f7fa,stroke:#00acc1,color:#006064;\n";
  // out << "  classDef output fill:#fce4ec,stroke:#d81b60,color:#880e4f;\n";
}

void Graph::Run() {
  LOG(INFO) << "Begin Dummy Run...";
  for (const auto & node : nodes_) {
    LOG(INFO) << "Excute node: " << node->name << " [" << node->op_type << "]";
    for (const auto & in : node->inputs) {
      LOG(INFO) << "    consume: " << in;
    }
    for (const auto & out : node->outputs) {
      LOG(INFO) << "    produce: " << out;
    }
  }
  LOG(INFO) << "Dummy Run Complete!";
}

std::vector<std::shared_ptr<Node>> Graph::TopoSort() {
  std::unordered_map<std::string, int> in_degree;
  std::unordered_map<std::string, std::vector<std::shared_ptr<Node>>> consumers;
  std::unordered_map<std::string, std::shared_ptr<Node>> node_map;

  // 初始化
  for (auto& node : nodes_) {
    in_degree[node->name] = 0;
    node_map[node->name] = node;
  }

  for (auto& node : nodes_) {
    for (const auto& input_name : node->inputs) {
      for (auto& other : nodes_) {
        if (std::find(other->outputs.begin(), other->outputs.end(), input_name) != other->outputs.end()) {
          consumers[other->name].push_back(node); 
          in_degree[node->name]++;
        }
      }
    }
  }

  std::queue<std::shared_ptr<Node>> q;
  std::vector<std::shared_ptr<Node>> sorted;

  for (const auto& [name, deg] : in_degree) {
    if (deg == 0) {
      q.push(node_map[name]);
    }
  }

  while (!q.empty()) {
    auto current = q.front();
    q.pop();
    sorted.push_back(current);
    for (const auto& next : consumers[current->name]) {
      in_degree[next->name]--;
      if (in_degree[next->name] == 0) {
        q.push(next);
      }
    }
  }

  if (sorted.size() != nodes_.size()) {
    LOG(FATAL) << "Graph has cycle or is malformed";
  }

  return sorted;
}

void Graph::Forward() {
  auto orderd_nodes = TopoSort();
  for (const auto& node : orderd_nodes) {
    //TODO: implement forward
    // node->forward(tensor_map_);
  }
}
}