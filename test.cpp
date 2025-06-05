#include <fstream>
#include <iostream>
#include "onnx/onnx.pb.h"

int main() {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::ifstream input("/Users/mizuiro/dev/mini_infer/model/mlp/mini_mlp.onnx", std::ios::in | std::ios::binary);
    if (!input) {
        std::cerr << "Cannot open model.onnx!\n";
        return 1;
    }

    onnx::ModelProto model;
    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Failed to parse model.onnx!\n";
        return 1;
    }

    std::cout << "Model parsed successfully!\n";
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
