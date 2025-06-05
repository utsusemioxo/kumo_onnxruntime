cmake_minimum_required(VERSION 3.14)

project(onnx_infer_minial)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g")

find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
find_package(benchmark REQUIRED)
find_package(glog REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/third_party)

add_executable(onnx_infer
  benchmark_runner.cpp
  graph.cpp
  attribute.cpp
  # executor.cpp
  tensor.cpp
  third_party/onnx/onnx.pb.cc
  # test.cpp
)

target_link_libraries(onnx_infer
  ${Protobuf_LIBRARIES}
  benchmark::benchmark
  glog::glog
  absl::strings
  absl::log_internal_check_op
  absl::log
)