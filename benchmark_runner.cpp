#include <benchmark/benchmark.h>
#include "graph.h"
#include "executor.h"
#include <fstream>
#include <glog/logging.h>
#include <memory>
#include "tensor.h"
#include <onnx/onnx.pb.h>

static std::unique_ptr<kumo::Graph> global_graph;

static void BM_GraphLoadAndRun(benchmark::State& state) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  for (auto _ : state) {
    if (!global_graph) {
      global_graph = std::make_unique<kumo::Graph>();
      bool success = global_graph->LoadFromONNX(
        "/Users/mizuiro/dev/mini_infer/model/mlp/mini_mlp.onnx");
      if (!success) {
        LOG(FATAL) << "Failed to load model.";
      }
    }

    global_graph->PrintGraph();

    std::ofstream out("graph.mmd");
    global_graph->ExportToMermaid(out);
    // global_graph->Run();

    google::protobuf::ShutdownProtobufLibrary();
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_GraphLoadAndRun)->Iterations(1);
BENCHMARK_MAIN();