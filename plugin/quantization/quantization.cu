#include "quantization-inl.h"
namespace mxnet {
namespace op {
template <>
Operator* CreateOp<gpu>(QuantiParam param, int dtype) {
  Operator* op = new QuantiOp<gpu, float>(param);
  return op;
}
}  // namespace op
}  // namespace mxnet