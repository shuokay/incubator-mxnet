#include "./quantization-inl.h"
namespace mxnet {
namespace op {
NNVM_REGISTER_OP(Quanti).set_attr<FCompute>("FCompute<gpu>", QuantizationCompute<gpu>);
NNVM_REGISTER_OP(_backward_Quanti).set_attr<FCompute>("FCompute<gpu>", QuantizationGrad<gpu>);
}  // namespace op
}  // namespace mxnet