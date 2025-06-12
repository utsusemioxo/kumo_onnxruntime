#include "graph.h"
#include "attribute.h"
#include "node.h"
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <onnx/onnx.pb.h>
#include <memory>
#include <stdexcept>
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

  // 1. Parse initializer -> TensorMap
  for (const auto &initializer : graph_proto.initializer()) {
    auto tensor = std::make_shared<Tensor>();
    LOG(INFO) << initializer.name();
    tensor->name = initializer.name();
    tensor->shape = { initializer.dims().begin(), initializer.dims().end()};
    tensor->dataType = onnx::TensorProto_DataType_Name(initializer.data_type());


    if (initializer.data_type() == onnx::TensorProto::FLOAT) {
      if (initializer.has_raw_data()) {
        LOG(INFO) << "parse_raw_data byte size=" << initializer.raw_data().size();
        const std::string& raw_data = initializer.raw_data();
        size_t num_elements = 1;
        for (int i = 0; i < initializer.dims_size(); ++i) {
          num_elements *= initializer.dims(i);
        }

        // size check for safety
        if (raw_data.size() != num_elements * sizeof(float)) {
          throw std::runtime_error("raw_data size mismatch with tensor shape");
        }

        tensor->floatData.resize(num_elements);
        std::memcpy(tensor->floatData.data(), raw_data.data(), raw_data.size());

      } else if (initializer.float_data_size() > 0) {
        LOG(INFO) << "parse float_data elem size=" << initializer.float_data_size();
        tensor->floatData = {initializer.float_data().begin(), initializer.float_data().end()};

      } else {
        LOG(INFO) << "parse data type not support";
      }
    }

    this->tensor_map_[tensor->name] = tensor;
  }

  // 2. parse input/output name
  for (const auto &input : graph_proto.input()) {
    this->input_names_.push_back(input.name());
  }
  
  for (const auto &output : graph_proto.output()) {
    this->output_names_.push_back(output.name());
  }

  // 3. parse nodes
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
    
    const std::string& op_type = node->op_type;
    LOG(INFO) << op_type;
  }
}
}