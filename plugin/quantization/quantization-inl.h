#ifndef MXNET_PLUGIN_QUANTIZATION_QUANTIZATION_INL_
#define MXNET_PLUGIN_QUANTIZATION_QUANTIZATION_INL_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>

#include <string>
#include <vector>

#include "../../src/operator/mshadow_op.h"
#include "../../src/operator/operator_common.h"

namespace mxnet {
namespace op {
namespace quanti {
enum QuantiOpInput { kData };
enum QuantiOpOutput { kOut };
}  // namespace quanti

struct QuantiParam : dmlc::Parameter<QuantiParam> {
  float scale;
  DMLC_DECLARE_PARAMETER(QuantiParam) {
    DMLC_DECLARE_FIELD(scale).describe("the scale of quantization");
  }
};
template <typename xpu>
static void QuantizationCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const auto& param = dmlc::get<QuantiParam>(attrs.parsed);
  using DType = float;
  DType qmin = -127, qmax = 127;
  Tensor<xpu, 4, DType> data = inputs[quanti::kData].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> out = outputs[quanti::kOut].get<xpu, 4, DType>(s);
  DType scale = param.scale;
  out = F<mshadow_op::minimum>(F<mshadow_op::maximum>(F<mshadow_op::round>(scale * data), qmin),
                               qmax) /
        scale;
}

template <typename xpu>
static void QuantizationGrad(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  using DType=float;
  Tensor<xpu, 4, DType> dgrad = outputs[quanti::kData].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> ograd = inputs[quanti::kOut].get<xpu, 4, DType>(s);
  Assign(dgrad, req[quanti::kData], ograd);
}
}  // namespace op
}  // namespace mxnet
#endif