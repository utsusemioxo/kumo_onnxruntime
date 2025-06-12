#include "graph_to_mlir_builder.h"

int main() {
  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::func::FuncDialect>();

  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  kumo::GraphDef graph;
  graph.LoadOnnxModelAsGraphDef("/Users/mizuiro/dev/mini_infer/model/mlp/mini_mlp.onnx");
  kumo::GraphToMlirBuilder builder(context, graph);
  mlir::ModuleOp module = builder.buildModule();

  module->dump();
  return 0;
}