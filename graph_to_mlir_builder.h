#include "graph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <iostream>
#include <vector>

#include <onnx/onnx.pb.h>

namespace kumo {

struct TensorDef {
  std::string name;
  std::vector<int64_t> shape;
};

struct NodeDef {
  std::string op;
  std::vector<std::string> inputs;
  std::string output;
  std::unordered_map<std::string, float> floatAttrs;
  std::unordered_map<std::string, int> intAttrs;
};

struct GraphDef {
  std::vector<TensorDef> inputs;
  std::vector<NodeDef> nodes;
  std::vector<TensorDef> outputs;

  void LoadOnnxModelAsGraphDef(const std::string &onnxPath);
};

class GraphToMlirBuilder {
public:
  GraphToMlirBuilder(mlir::MLIRContext &context, const GraphDef &graph);
  mlir::ModuleOp buildModule();
private:
  mlir::RankedTensorType getTensorType(mlir::ArrayRef<int64_t> shape);
  mlir::RankedTensorType getTensorTypeFromValue(mlir::Value lhs, mlir::Value rhs);

  mlir::MLIRContext &context;
  mlir::OpBuilder builder;
  mlir::Location location;
  mlir::ModuleOp module;
  const GraphDef &graph;
  std::unordered_map<std::string, mlir::Value> valueMap;
};
} // namespace kumo