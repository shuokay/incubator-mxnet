#include "../../src/operator/elemwise_op_common.h"
#include "./quantization-inl.h"
namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(QuantiParam);
NNVM_REGISTER_OP(Quanti)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<QuantiParam>)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>({"data"});
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<-1, 1>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, 1>)
    .set_attr<FCompute>("FCompute<cpu>", QuantizationCompute<cpu>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_Quanti"})
    .add_argument("data", "NDArray-or-Symbol", "")
    .add_arguments(QuantiParam::__FIELDS__());
NNVM_REGISTER_OP(_backward_Quanti)
    .set_attr_parser(ParamParser<QuantiParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", QuantizationGrad<cpu>);
}  // namespace op
}  // namespace mxnet