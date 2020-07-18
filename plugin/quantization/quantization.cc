#include "quantization-inl.h"
namespace mxnet {
namespace op {
template <>
Operator* CreateOp<cpu>(QuantiParam param, int dtype) {
  Operator* op = new QuantiOp<cpu, float>(param);
  return op;
}

Operator* QuantiProp::CreateOperatorEx(Context ctx,
                                       std::vector<TShape>* in_shape,
                                       std::vector<int>* in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(QuantiParam);
MXNET_REGISTER_OP_PROPERTY(Quanti, QuantiProp)
    .add_argument("data", "Symbol", "Input data to the Quanti")
    .add_arguments(QuantiParam::__FIELDS__())
    .describe("Quanti");
}  // namespace op
}  // namespace mxnet