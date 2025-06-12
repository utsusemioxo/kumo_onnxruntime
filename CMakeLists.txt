cmake_minimum_required(VERSION 3.14)

project(onnx_infer_minial)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
find_package(benchmark REQUIRED)
find_package(glog REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/third_party
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

add_definitions(${LLVM_DEFINITIONS})

add_executable(onnx_infer
  benchmark_runner.cpp
  graph.cpp
  attribute.cpp
  # executor.cpp
  tensor.cpp
  third_party/onnx/onnx.pb.cc
  # test_onnx_pb.cpp
  # test_onnx_tosa.cpp
  # graph_to_mlir_builder.cpp
)

set(MLIR_LIBS
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRBytecodeReader
  MLIRAsmParser
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRFunctionInterfaces
  MLIRPresburger
  MLIRSideEffectInterfaces
  MLIRViewLikeInterface
  MLIRFuncDialect
  MLIRTransforms
  MLIRTosaDialect
)

target_link_libraries(onnx_infer
  PRIVATE
  ${Protobuf_LIBRARIES}
  benchmark::benchmark
  glog::glog
  absl::strings
  absl::log_internal_check_op
  absl::log
  ${MLIR_LIBS}
  ${LLVM_LIBS}
)