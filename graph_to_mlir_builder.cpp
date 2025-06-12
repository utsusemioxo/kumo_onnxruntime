#include "graph_to_mlir_builder.h"
#include "graph.h"
#include "onnx/onnx.pb.h"
#include <cstdint>
#include <stdexcept>
#include <string>


namespace kumo {
using namespace mlir;
GraphToMlirBuilder::GraphToMlirBuilder(MLIRContext &context, const GraphDef &grpah)
    : builder(&context), graph(grpah), context(context),
      location(builder.getUnknownLoc()) {}

ModuleOp GraphToMlirBuilder::buildModule() {
  module = ModuleOp::create(location);

  // Build func signature
  SmallVector<Type> inputTypes;
  for (auto &tensor : graph.inputs) {
    inputTypes.push_back(getTensorType(tensor.shape));
  }

  SmallVector<Type> outputTypes;
  for (auto &tensor : graph.outputs) {
    outputTypes.push_back(getTensorType(tensor.shape));
  }

  auto funcType = builder.getFunctionType(inputTypes, outputTypes);
  auto funcOp = builder.create<func::FuncOp>(location, "main", funcType);

  module.push_back(funcOp);

  // Build body
  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  for (size_t i = 0; i < graph.inputs.size(); ++i) {
    valueMap[graph.inputs[i].name] = entryBlock->getArgument(i);
  }

  for (const auto &node : graph.nodes) {
    // todo: support ops
    if (node.op == "Gemm") {
      // Y = alpha * A * B + beta * C
      // A = inputs[0], B = inputs[1], C = inputs[2] (bias)
      // alpha, beta = scalae (常为 1.0)
      // transA / transB = 是否转置 A/B
      auto A = valueMap[node.inputs[0]];
      auto B = valueMap[node.inputs[1]];
      auto C = valueMap[node.inputs[2]]; // bias
    
      int transA = node.intAttrs.count("transA") ? node.intAttrs.at("transA") : 0;
      int transB = node.intAttrs.count("transB") ? node.intAttrs.at("transB") : 0;
      float alpha = node.floatAttrs.count("alpha") ? node.floatAttrs.at("alpha") : 1.0f;
      float beta = node.floatAttrs.count("beta") ? node.floatAttrs.at("beta") : 1.0f;
    
      // 先对 A/B 转置，如果需要
      if (transA) {
        // A = builder.create<tosa::TransposeOp>(
        //   location,
        //   A.getType(),
        //   A,
        //   createTrans
        // );
      }
    
      // // 如果 alpha ≠ 1，乘一下
      // if (alpha != 1.0f) {
      //   A = builder.create<tosa::MulOp>(location, A.getType(), A, createConstScalar(builder, location, alpha));
      // }
    
      // auto matmulResultType = getTensorTypeFromValue(A, B); // 你已有的 helper
      // auto matmul = builder.create<tosa::MatMulOp>(location, matmulResultType, A, B);
    
      // auto result = matmul.getResult();
    
      // // 如果 beta ≠ 1，乘一下 bias
      // if (beta != 1.0f) {
      //   C = builder.create<tosa::MulOp>(location, C.getType(), C, createConstScalar(builder, location, beta));
      // }
    
      // // 加 bias
      // auto finalResult = builder.create<tosa::AddOp>(location, matmulResultType, result, C);
    
      // valueMap[node.output] = finalResult.getResult();
    } else if (node.op == "Relu") {
      auto input = valueMap[node.inputs[0]];
      auto resultType = input.getType();
      auto op = builder.create<tosa::ClampOp>(
        location,
        resultType,
        input
      );
      valueMap[node.output] = op.getResult();
    }
  }

  // Return output
  SmallVector<Value> results;
  for (auto &tensor : graph.outputs) {
    results.push_back(valueMap[tensor.name]);
  }

  {
    for (auto &tensor : graph.outputs) {
      auto it = valueMap.find(tensor.name);
      if (it == valueMap.end()) {
        llvm::errs() << "Error: tensor name " << tensor.name << " not found in valueMap!\n";
        abort();
      }
      if (!it->second) {
        llvm::errs() << "Error: valueMap[" << tensor.name << "] is null!\n";
        abort();
      }
      results.push_back(it->second);
    }
    
  }
  builder.create<func::ReturnOp>(location, results);

  return module;
}

RankedTensorType GraphToMlirBuilder::getTensorType(ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, builder.getF32Type());
}

RankedTensorType GraphToMlirBuilder::getTensorTypeFromValue(Value lhs, Value rhs) {
  auto lhsTy = mlir::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsTy = mlir::dyn_cast<RankedTensorType>(rhs.getType());
  return RankedTensorType::get({lhsTy.getShape()[0], rhsTy.getShape()[1]}, builder.getF32Type());
}


void GraphDef::LoadOnnxModelAsGraphDef(const std::string &onnxPath) {

  // 1. read ONNX file
  onnx::ModelProto model;
  std::ifstream input(onnxPath, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open ONNX file: " + onnxPath);
  }

  if (!model.ParseFromIstream(&input)) {
    throw std::runtime_error("Failed to parse ONNX file.");
  }

  const auto &graph = model.graph();

  // 2. get inputs
  for (const auto &input : graph.input()) {
    if (!input.has_type() || !input.type().has_tensor_type()) {
      continue;
    }

    TensorDef t;
    t.name = input.name();
    const auto &tensorType = input.type().tensor_type();
    if (tensorType.has_shape()) {
      for (const auto &dim : tensorType.shape().dim()) {
        t.shape.push_back(dim.dim_value());
      }
    }
    inputs.push_back(t);
  }

  // 3. get outputs
  for (const auto &output : graph.output()) {
    if (!output.has_type() || !output.type().has_tensor_type()) {
      continue;
    }

    TensorDef t;
    t.name = output.name();
    const auto &tensorType = output.type().tensor_type();
    if (tensorType.has_shape()) {
      for (const auto &dim : tensorType.shape().dim()) {
        t.shape.push_back(dim.dim_value());
      }
    }
    outputs.push_back(t);
  }

  // 4. get nodes
  for (const auto &node : graph.node()) {
    NodeDef def;
    def.op = node.op_type();

    for (const auto &inputName : node.input()) {
      def.inputs.push_back(inputName);
    }

    if (node.output_size() != 1) {
      throw std::runtime_error("Currently only single-output nodes are supported: " + def.op);
    }

    for (const auto &attr : node.attribute()) {
      const std::string& name = attr.name();
      if (attr.has_f()) {
        float fval = attr.f();
        std::cout << "Float attribute: " << name << " = " << fval << std::endl;
      } else if (attr.has_i()) {
        int64_t ival = attr.i();
        std::cout << "Int attribute: " << name << " = " << ival << std::endl;
      } else if (attr.has_s()) {
        std::string sval = attr.s();
        std::cout << "String attribute: " << name << " = " << sval << std::endl;
      } else if (attr.floats_size() > 0) {
        std::cout << "Float list attribute: " << name << std::endl;
        for (const auto f : attr.floats()) {
          std::cout << "  - " << f << std::endl; 
        }
      } else if (attr.ints_size() > 0) {
        std::cout << "Int list attribute: " << name << std::endl;
        for (auto i : attr.ints()) {
          std::cout << "  - " << i << std::endl;
        }
      } else {
        std::cout << "Unknown attribute type: " << name << std::endl;
      }
    }
  
    def.output = node.output(0);
    nodes.push_back(def);
  }
}

} // namespace kumo