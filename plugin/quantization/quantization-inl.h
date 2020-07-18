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

namespace mshadow_op {
struct clip_ex {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType v, DType min, DType max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
  }
};
}  // namespace mshadow_op

template <typename xpu, typename DType>
class QuantiOp : public Operator {
 public:
  explicit QuantiOp(QuantiParam param) { this->param_ = param; }
  virtual void Forward(const OpContext& ctx,
                       const std::vector<TBlob>& in_data,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& out_data,
                       const std::vector<TBlob>& aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu>* s = ctx.get_stream<xpu>();
    DType qmin = -127, qmax = 127;
    Tensor<xpu, 4, DType> data = in_data[quanti::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[quanti::kOut].get<xpu, 4, DType>(s);
    DType scale = param_.scale;
    out = F<mshadow_op::minimum>(F<mshadow_op::maximum>(F<mshadow_op::round>(scale * data), qmin),
                                 qmax) /
          scale;
  }
  virtual void Backward(const OpContext& ctx,
                        const std::vector<TBlob>& out_grad,
                        const std::vector<TBlob>& in_data,
                        const std::vector<TBlob>& out_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& in_grad,
                        const std::vector<TBlob>& aux_states) {
    using namespace mshadow;
    Stream<xpu>* s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> dgrad = in_grad[quanti::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> ograd = out_grad[quanti::kOut].get<xpu, 4, DType>(s);
    Assign(dgrad, req[quanti::kData], ograd);
  }

 private:
  QuantiParam param_;
};

template <typename xpu>
Operator* CreateOp(QuantiParam param, int dtype);

#ifdef DMLC_USE_CXX11
class QuantiProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string>>& kwargs) override {
    param_.Init(kwargs);
  }
  std::map<std::string, std::string> GetParams() const override { return param_.__DICT__(); }
  bool InferShape(std::vector<TShape>* in_shape,
                  std::vector<TShape>* out_shape,
                  std::vector<TShape>* aux_shape) const override {
    SHAPE_ASSIGN_CHECK(*out_shape, quanti::kOut, (*in_shape)[quanti::kData]);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new QuantiProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "Quanti"; }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx,
                             std::vector<TShape>* in_shape,
                             std::vector<int>* in_type) const override;

 private:
  QuantiParam param_;
};
#endif

}  // namespace op
}  // namespace mxnet

#endif