#pragma once
#include "graph.h"
#include "tensor.h"

namespace kumo {
class Executor {
public:
  Tensor Run(const Graph& graph, const Tensor& input);
};
}