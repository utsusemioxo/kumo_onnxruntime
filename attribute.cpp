#include "attribute.h"
#include "onnx/onnx.pb.h"

namespace kumo {
Attribute ConvertAttribute(const onnx::AttributeProto& onnx_attr) {
  Attribute attr;
  attr.name = onnx_attr.name();

  switch (onnx_attr.type()) {
    case onnx::AttributeProto::INT:
    {
      attr.type = Attribute::Type::INT;
      attr.i = onnx_attr.i();
      break;
    }
    case onnx::AttributeProto::FLOAT:
    {
      attr.type = Attribute::Type::FLOAT;
      attr.f = onnx_attr.f();
      break;
    }
    case onnx::AttributeProto::STRING:
    {
      attr.type = Attribute::Type::STRING;
      attr.s = onnx_attr.s();
      break;
    }
    case onnx::AttributeProto::INTS:
    {
      attr.type = Attribute::Type::INTS;
      attr.ints.assign(onnx_attr.ints().begin(), onnx_attr.ints().end());
      break;
    }
    case onnx::AttributeProto::FLOATS:
    {
      attr.type = Attribute::Type::FLOATS;
      attr.floats.assign(onnx_attr.floats().begin(), onnx_attr.floats().end());
      break;
    }
    case onnx::AttributeProto::STRINGS:
    {
      attr.type = Attribute::Type::STRINGS;
      for (const auto& s : onnx_attr.strings()) {
        attr.strings.push_back(s);
      }
      break;
    }
    default:
    {
      attr.type = Attribute::Type::UNDEFINED;
      break;
    }
  }
  return attr;
}
}