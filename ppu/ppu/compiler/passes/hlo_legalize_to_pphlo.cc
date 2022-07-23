// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// This file implements logic for lowering HLO dialect to pphlo dialect.

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ppu/compiler/dialect/scfhlo/IR/scfhlo_ops.h"
#include "ppu/compiler/passes/map_mhlo_to_pphlo_op.h"
#include "ppu/compiler/passes/pass_details.h"
#include "ppu/compiler/passes/value_visibility_map.h"
#include "ppu/compiler/passes/visibility_inference.h"
#include "ppu/dialect/pphlo_base_enums.h"
#include "ppu/utils/exception.h"

namespace mlir::pphlo {
namespace {

class ConversionTypeTools final : public TypeTools {
public:
  ~ConversionTypeTools() override = default;

  bool isUnknownType(const Type &t) const {
    if (const auto &rt = t.dyn_cast<RankedTensorType>()) {
      return isUnknownType(rt.getElementType());
    }
    return t.isa<UIntegerType>() || t.isa<UFixedpointType>();
  }

private:
  bool isIntegerScalar(const Type &t) const override {
    return t.isa<PIntegerType>() || t.isa<SIntegerType>() ||
           t.isa<UIntegerType>();
  }

  bool isFixedpointScalar(const Type &t) const override {
    return t.isa<PFixedpointType>() || t.isa<SFixedpointType>() ||
           t.isa<UFixedpointType>();
  }
};

/// This struct carries information of io visibility
struct IoVisibilityInfo {
  std::vector<Visibility> inputs;

  void convertFromStrings(llvm::ArrayRef<std::string> data) {
    for (const auto &s : data) {
      const auto symbolized = symbolizeEnum<Visibility>(s);
      PPU_ENFORCE(symbolized.hasValue());
      inputs.emplace_back(*symbolized);
    }
  }

  Visibility getInputVisibility(size_t idx) const {
    if (idx >= inputs.size()) {
      return Visibility::VIS_PUBLIC;
    }
    return inputs[idx];
  }
};

ValueVisibilityMap VisibilityDiscovery(ModuleOp op,
                                       const IoVisibilityInfo &input_vis) {
  // Get the main function
  auto entry_func = op.lookupSymbol<FuncOp>("main");

  PPU_ENFORCE(entry_func != nullptr);

  ValueVisibilityMap vis_map;
  // Populate top level io visibility
  for (const auto &blockargs : entry_func.getBody().getArguments()) {
    vis_map.setValueVisibility(
        blockargs, input_vis.getInputVisibility(blockargs.getArgNumber()));
  }

  VisibilityInference inference(vis_map);
  inference.inferFunc(entry_func);

  return vis_map;
}

ConversionTypeTools typetools_;
/// Type converter for mhlo type to pphlo types
class HloToPPHloTypeConverter : public TypeConverter {
private:
  Type convertRankedTensorType(RankedTensorType type) {
    Type oriElmTy = type.getElementType();
    Type newElmTy;
    if (oriElmTy.isa<::mlir::FloatType>()) {
      newElmTy = ::mlir::pphlo::UFixedpointType::get(type.getContext());
    } else if (oriElmTy.isa<::mlir::IntegerType>()) {
      newElmTy = ::mlir::pphlo::UIntegerType::get(type.getContext());
    } else {
      newElmTy = oriElmTy;
    }
    return RankedTensorType::get(type.getShape(), newElmTy);
  }

  static Value materializeToMPCTensor(OpBuilder &builder, RankedTensorType type,
                                      ValueRange inputs, Location loc) {
    PPU_ENFORCE(inputs.size() == 1);
    PPU_ENFORCE(inputs[0].getType().isa<RankedTensorType>());

    // To unknown type is always a noop, just forward operands
    if (typetools_.isUnknownType(type)) {
      return inputs.front();
    }

    // Deferred materialization
    auto op = builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0]);

    return op.getResults()[0];
  }

public:
  HloToPPHloTypeConverter() {
    // Keep all types unchanged.
    addConversion([&](RankedTensorType type) -> Type {
      return convertRankedTensorType(type);
    });
    addTargetMaterialization(materializeToMPCTensor);
  }

  static Type getTypeWithVisibility(Type type, Visibility vis) {
    switch (vis) {
    case Visibility::VIS_PUBLIC:
      return typetools_.toPublicType(type);
    case Visibility::VIS_SECRET:
      return typetools_.toSecretType(type);
    }
    llvm_unreachable("Should not reach here.");
  }
};

Visibility getOperandVisibility(const mlir::Value &v) {
  if (typetools_.isUnknownType(v.getType())) {
    if (auto dop = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      for (const auto &result : llvm::enumerate(dop.getResults())) {
        if (result.value() == v) {
          return typetools_.getTypeVisibility(
              dop->getOperandTypes()[result.index()]);
        }
      }
    }
    llvm_unreachable("Should not hit here.");
  }
  return typetools_.getTypeVisibility(v.getType());
}

class FuncOpConverter : public OpConversionPattern<::mlir::FuncOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  FuncOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                  const ValueVisibilityMap &vis)
      : OpConversionPattern<::mlir::FuncOp>(typeConverter, context), vis_(vis) {
  }

  LogicalResult
  matchAndRewrite(::mlir::FuncOp op, ::mlir::FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);

    auto functionType = op.getType();
    auto &region = op.getBody();
    // Convert non-entry blocks
    SmallVector<TypeConverter::SignatureConversion, 2> conversions;
    for (Block &block : llvm::drop_begin(region, 1)) {
      conversions.emplace_back(block.getNumArguments());
      TypeConverter::SignatureConversion &back = conversions.back();
      for (BlockArgument blockArgument : block.getArguments()) {
        auto idx = blockArgument.getArgNumber();
        auto vis_v = vis_.getValueVisibility(blockArgument);
        auto convertedType = HloToPPHloTypeConverter::getTypeWithVisibility(
            typeConverter->convertType(blockArgument.getType()), vis_v);
        back.addInputs(idx, convertedType);
      }
    }
    if (failed(rewriter.convertNonEntryRegionTypes(&region, *typeConverter,
                                                   conversions))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(functionType.getNumInputs());
    for (const auto &blockarg : llvm::enumerate(op.getBody().getArguments())) {
      auto vis_v = vis_.getValueVisibility(blockarg.value());
      auto convertedType = HloToPPHloTypeConverter::getTypeWithVisibility(
          typeConverter->convertType(blockarg.value().getType()), vis_v);
      conversion.addInputs(blockarg.index(), convertedType);
    }

    // If the SignatureConversion doesn't apply, bail out.
    if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter(),
                                           &conversion))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    // Update the signature of the function.
    SmallVector<Type, 2> newResultTypes;
    if (failed(typeConverter->convertTypes(functionType.getResults(),
                                           newResultTypes))) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }

    // Update return types
    auto retOp = llvm::dyn_cast<::mlir::ReturnOp>(op.getBody().back().back());
    PPU_ENFORCE(retOp->getNumOperands() == newResultTypes.size());

    for (const auto &resultType : llvm::enumerate(newResultTypes)) {
      auto vis_v =
          vis_.getValueVisibility(retOp.getOperand(resultType.index()));
      newResultTypes[resultType.index()] =
          HloToPPHloTypeConverter::getTypeWithVisibility(resultType.value(),
                                                         vis_v);
    }
    op.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                        newResultTypes));
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

// TODO: Add a rsqrt op if we have cases that can benefit from fused op.
class RSqrtOpConverter : public OpConversionPattern<mhlo::RsqrtOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  RSqrtOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                   const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::RsqrtOp>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::RsqrtOp op, mhlo::RsqrtOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    OpBuilder builder(op);

    auto r = builder.create<pphlo::SqrtOp>(op->getLoc(), resultType,
                                           adaptor.getOperands());

    rewriter.replaceOpWithNewOp<pphlo::ReciprocalOp>(op, resultType, r);

    return success();
  }
};

class ReturnOpConverter : public OpConversionPattern<::mlir::ReturnOp> {
public:
  ReturnOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                    const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<::mlir::ReturnOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(::mlir::ReturnOp op, ::mlir::ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *operation = op.getOperation();
    rewriter.updateRootInPlace(
        op, [&]() { operation->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class HloCompToPPHloOpConverter : public OpConversionPattern<mhlo::CompareOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloCompToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                            const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::CompareOp>(typeConverter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::CompareOp hloOp, mhlo::CompareOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hloOp.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hloOp.getType()), result_vis);

    auto comp_direction = hloOp.comparison_direction();

    SmallVector<Value, 2> operands(adaptor.getOperands());

    if (comp_direction == mhlo::stringifyEnum(mhlo::ComparisonDirection::EQ)) {
      rewriter.replaceOpWithNewOp<pphlo::EqualOp>(hloOp, resultType, operands);
    } else if (comp_direction ==
               mhlo::stringifyEnum(mhlo::ComparisonDirection::NE)) {
      rewriter.replaceOpWithNewOp<pphlo::NotEqualOp>(hloOp, resultType,
                                                     operands);
    } else if (comp_direction ==
               mhlo::stringifyEnum(mhlo::ComparisonDirection::LT)) {
      rewriter.replaceOpWithNewOp<pphlo::LessOp>(hloOp, resultType, operands);
    } else if (comp_direction ==
               mhlo::stringifyEnum(mhlo::ComparisonDirection::LE)) {
      rewriter.replaceOpWithNewOp<pphlo::LessEqualOp>(hloOp, resultType,
                                                      operands);
    } else if (comp_direction ==
               mhlo::stringifyEnum(mhlo::ComparisonDirection::GT)) {
      rewriter.replaceOpWithNewOp<pphlo::GreaterOp>(hloOp, resultType,
                                                    operands);
    } else if (comp_direction ==
               mhlo::stringifyEnum(mhlo::ComparisonDirection::GE)) {
      rewriter.replaceOpWithNewOp<pphlo::GreaterEqualOp>(hloOp, resultType,
                                                         operands);
    } else {
      return failure();
    }
    return success();
  }
};

template <typename HloReduceOpTy>
struct ReduceOpConverter : public OpConversionPattern<HloReduceOpTy> {
private:
  const ValueVisibilityMap &vis_;

public:
  ReduceOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                    const ValueVisibilityMap &vis)
      : OpConversionPattern<HloReduceOpTy>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(HloReduceOpTy op,
                  typename ReduceOpConverter::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // We may need to materialize operands
    llvm::SmallVector<Value> materialized_operands;
    llvm::SmallVector<Type> result_types;
    size_t num_results = op.getNumResults();

    materialized_operands.resize(2 * num_results);
    result_types.resize(num_results);

    llvm::SmallVector<Value, 4> operands(adaptor.getOperands());

    OpBuilder builder(op);

    auto materialize = [&, this](size_t idx, Visibility result_vis) {
      auto current_vis = getOperandVisibility(operands[idx]);
      if (result_vis == current_vis) {
        materialized_operands[idx] = operands[idx];
      } else {
        auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
            operands[idx].getType(), result_vis);
        materialized_operands[idx] =
            this->getTypeConverter()->materializeTargetConversion(
                builder, op.getLoc(), new_type, operands[idx]);
      }
    };

    for (size_t idx = 0; idx < num_results; ++idx) {
      auto result_vis = vis_.getValueVisibility(op.getResult(idx));
      // Check input vis
      materialize(idx, result_vis);
      materialize(idx + num_results, result_vis);
      // Push result type
      result_types[idx] = HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(op.getType(idx)), result_vis);
    }

    // Convert the region signature.
    auto &entry_block = op.body().front();
    TypeConverter::SignatureConversion sig_conversion(
        entry_block.getNumArguments());

    for (const auto &arg : entry_block.getArguments()) {
      auto arg_t = this->getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    auto new_op =
        rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<HloReduceOpTy>>(
            op, result_types, materialized_operands, op->getAttrs());

    // Copy over the operations inside the region.
    rewriter.inlineRegionBefore(op.body(), new_op.body(), new_op.body().end());

    if (failed(rewriter.convertRegionTypes(
            &new_op.body(), *this->getTypeConverter(), &sig_conversion))) {
      return failure();
    }

    return success();
  }
};

struct IfOpConverter : public OpConversionPattern<scfhlo::IfOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  IfOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                const ValueVisibilityMap &vis)
      : OpConversionPattern<scfhlo::IfOp>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(scfhlo::IfOp op, scfhlo::IfOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<Type, 4> resultTypes;
    {
      for (const auto &ret : op->getResults()) {
        auto result_vis = vis_.getValueVisibility(ret);
        resultTypes.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
            this->getTypeConverter()->convertType(ret.getType()), result_vis));
      }
    }

    // Convert true region signature.
    auto &true_region = op.true_branch();
    TypeConverter::SignatureConversion true_sig_conversion(
        true_region.getNumArguments());

    for (const auto &arg : true_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      true_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Convert false region signature.
    auto &false_region = op.false_branch();
    TypeConverter::SignatureConversion false_sig_conversion(
        false_region.getNumArguments());

    for (const auto &arg : false_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      false_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    llvm::SmallVector<Value, 2> operands(adaptor.getOperands());

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::IfOp>(
        op, resultTypes, operands, op->getAttrs());

    // Copy over the operations inside true/false region.
    rewriter.inlineRegionBefore(op.true_branch(), new_op.true_branch(),
                                new_op.true_branch().end());
    rewriter.inlineRegionBefore(op.false_branch(), new_op.false_branch(),
                                new_op.false_branch().end());

    if (failed(rewriter.convertRegionTypes(&new_op.true_branch(),
                                           *getTypeConverter(),
                                           &true_sig_conversion))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&new_op.false_branch(),
                                           *getTypeConverter(),
                                           &false_sig_conversion))) {
      return failure();
    }

    return success();
  }
};

struct WhileOpConverter : public OpConversionPattern<mhlo::WhileOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  WhileOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                   const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::WhileOp>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::WhileOp op, mhlo::WhileOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type, 4> resultTypes;
    {
      for (const auto &ret : op->getResults()) {
        auto result_vis = vis_.getValueVisibility(ret);
        resultTypes.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
            this->getTypeConverter()->convertType(ret.getType()), result_vis));
      }
    }

    // Convert cond region signature.
    auto &cond_region = op.cond();
    TypeConverter::SignatureConversion cond_sig_conversion(
        cond_region.getNumArguments());

    for (const auto &arg : cond_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      cond_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // Convert body region signature.
    auto &body_region = op.body();
    TypeConverter::SignatureConversion body_sig_conversion(
        body_region.getNumArguments());

    for (const auto &arg : body_region.getArguments()) {
      auto arg_t = getTypeConverter()->convertType(arg.getType());
      auto lower_t = HloToPPHloTypeConverter::getTypeWithVisibility(
          arg_t, vis_.getValueVisibility(arg));
      body_sig_conversion.addInputs(arg.getArgNumber(), lower_t);
    }

    // May need to materialize operands
    OpBuilder builder(op);
    llvm::SmallVector<Value, 6> operands(adaptor.getOperands());
    llvm::SmallVector<Value, 6> materializedOperands;
    for (const auto &operand : llvm::enumerate(operands)) {
      auto currentVis = getOperandVisibility(operand.value());
      auto targetVis =
          vis_.getValueVisibility(op.body().getArgument(operand.index()));
      if (currentVis == targetVis) {
        materializedOperands.emplace_back(operand.value());
      } else {
        auto new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
            operand.value().getType(), targetVis);
        materializedOperands.emplace_back(
            getTypeConverter()->materializeTargetConversion(
                builder, op->getLoc(), new_type, operand.value()));
      }
    }

    auto new_op = rewriter.replaceOpWithNewOp<pphlo::WhileOp>(
        op, resultTypes, materializedOperands, op->getAttrs());

    // Copy over the operations inside body region.
    rewriter.inlineRegionBefore(op.body(), new_op.body(), new_op.body().end());
    rewriter.inlineRegionBefore(op.cond(), new_op.cond(), new_op.cond().end());

    if (failed(rewriter.convertRegionTypes(&new_op.body(), *getTypeConverter(),
                                           &body_sig_conversion))) {
      return failure();
    }

    if (failed(rewriter.convertRegionTypes(&new_op.cond(), *getTypeConverter(),
                                           &cond_sig_conversion))) {
      return failure();
    }

    return success();
  }
};

template <typename HloOpTy>
class HloToPPHloOpConverter : public OpConversionPattern<HloOpTy> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<HloOpTy>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(HloOpTy hloOp,
                  typename HloToPPHloOpConverter::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hloOp.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hloOp.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<HloOpTy>>(
        hloOp, resultType, adaptor.getOperands(), hloOp->getAttrs());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::ConstOp>
    : public OpConversionPattern<mhlo::ConstOp> {
public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<mhlo::ConstOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(mhlo::ConstOp hloOp, mhlo::ConstOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<mhlo::ConstOp>>(
        hloOp, hloOp.value());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::IotaOp>
    : public OpConversionPattern<mhlo::IotaOp> {
public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<mhlo::IotaOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(mhlo::IotaOp hloOp, mhlo::IotaOpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hloOp.getType()),
        Visibility::VIS_PUBLIC);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<mhlo::IotaOp>>(
        hloOp, resultType, hloOp.iota_dimension());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mlir::ConstantOp>
    : public OpConversionPattern<mlir::ConstantOp> {
public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<mlir::ConstantOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(mlir::ConstantOp op, mlir::ConstantOpAdaptor /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::ConstOp>(op, op.getValue());
    return success();
  }
};

/// Need a special conversion rule for Dot to drop precision configs
template <>
class HloToPPHloOpConverter<mhlo::DotOp>
    : public OpConversionPattern<mhlo::DotOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::DotOp>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::DotOp hloOp, mhlo::DotOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(hloOp.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(hloOp.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::HloToPPHloOp<mhlo::DotOp>>(
        hloOp, resultType, adaptor.getOperands());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::ReturnOp>
    : public OpConversionPattern<mhlo::ReturnOp> {
public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap & /*unused*/)
      : OpConversionPattern<mhlo::ReturnOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(mhlo::ReturnOp op, mhlo::ReturnOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<pphlo::ReturnOp>(op, llvm::None,
                                                 adaptor.getOperands());
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::RngUniformOp>
    : public OpConversionPattern<mhlo::RngUniformOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::RngUniformOp>(typeConverter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::RngUniformOp op, mhlo::RngUniformOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::RngUniformOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::SortOp>
    : public OpConversionPattern<mhlo::SortOp> {
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::SortOp>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::SortOp op, mhlo::SortOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto comp_ret =
        llvm::dyn_cast<mhlo::ReturnOp>(op.comparator().back().getTerminator());
    PPU_ENFORCE(comp_ret.getNumOperands() == 1,
                "SortOp comparator can only return one value");

    // Sanity comparator
    auto comp_op = comp_ret->getOperand(0).getDefiningOp<mhlo::CompareOp>();
    PPU_ENFORCE(comp_op != nullptr,
                "SortOp comparator must return directly from a CompareOp");

    bool is_less = (comp_op.comparison_direction() ==
                    mhlo::stringifyEnum(mhlo::ComparisonDirection::LT));

    PPU_ENFORCE(
        is_less || comp_op.comparison_direction() ==
                       mhlo::stringifyEnum(mhlo::ComparisonDirection::GT),
        "Expect comparator only contains less than or greater than comparison");

    // CompareOp operands must directly from first and second blkarg
    PPU_ENFORCE(comp_op.lhs() == op.comparator().getArgument(0),
                "CompareOp lhs must be blkarg(0)");
    PPU_ENFORCE(comp_op.rhs() == op.comparator().getArgument(1),
                "CompareOp rhs must be blkarg(1)");

    llvm::SmallVector<Type, 2> ret_types;
    for (const auto &ret : op->getResults()) {
      auto ret_vis = vis_.getValueVisibility(ret);
      ret_types.emplace_back(HloToPPHloTypeConverter::getTypeWithVisibility(
          this->getTypeConverter()->convertType(ret.getType()), ret_vis));
    }

    rewriter.replaceOpWithNewOp<pphlo::SortOp>(
        op, ret_types, adaptor.getOperands(), op.dimension(), op.is_stable(),
        is_less);

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::GatherOp>
    : public OpConversionPattern<mhlo::GatherOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::GatherOp>(typeConverter, context), vis_(vis) {
  }

  LogicalResult
  matchAndRewrite(mhlo::GatherOp op, mhlo::GatherOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto old_attr = op.dimension_numbers();
    pphlo::GatherDimensionNumbersAttr attr = GatherDimensionNumbersAttr::get(
        op.getContext(), old_attr.getOffsetDims(),
        old_attr.getCollapsedSliceDims(), old_attr.getStartIndexMap(),
        old_attr.getIndexVectorDim());

    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::GatherOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1],
        attr, op.slice_sizes(), op.indices_are_sorted());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::ConvOp>
    : public OpConversionPattern<mhlo::ConvOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::ConvOp>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::ConvOp op, mhlo::ConvOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto old_attr = op.dimension_numbers();
    auto attr = ConvDimensionNumbersAttr::get(
        op->getContext(), old_attr.getInputBatchDimension(),
        old_attr.getInputFeatureDimension(),
        old_attr.getInputSpatialDimensions(),
        old_attr.getKernelInputFeatureDimension(),
        old_attr.getKernelOutputFeatureDimension(),
        old_attr.getKernelSpatialDimensions(),
        old_attr.getOutputBatchDimension(),
        old_attr.getOutputFeatureDimension(),
        old_attr.getOutputSpatialDimensions());

    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    rewriter.replaceOpWithNewOp<pphlo::ConvOp>(
        op, resultType, adaptor.getOperands()[0], adaptor.getOperands()[1],
        op.window_strides().getValueOr(nullptr),
        op.padding().getValueOr(nullptr), op.lhs_dilation().getValueOr(nullptr),
        op.rhs_dilation().getValueOr(nullptr), attr, op.feature_group_count(),
        op.batch_group_count());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::PadOp>
    : public OpConversionPattern<mhlo::PadOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::PadOp>(typeConverter, context), vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::PadOp op, mhlo::PadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type result_type = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    llvm::SmallVector<Value, 2> materialized_operands;
    OpBuilder builder(op);
    for (const auto &old_operand : llvm::enumerate(op.getOperands())) {
      auto op_vis = vis_.getValueVisibility(old_operand.value());
      if (op_vis != result_vis) {
        Type new_type = HloToPPHloTypeConverter::getTypeWithVisibility(
            adaptor.getOperands()[old_operand.index()].getType(), result_vis);
        materialized_operands.emplace_back(
            getTypeConverter()->materializeTargetConversion(
                builder, op.getLoc(), new_type,
                adaptor.getOperands()[old_operand.index()]));
      } else {
        materialized_operands.emplace_back(
            adaptor.getOperands()[old_operand.index()]);
      }
    }

    rewriter.replaceOpWithNewOp<pphlo::PadOp>(
        op, result_type, materialized_operands, op->getAttrs());

    return success();
  }
};

template <>
class HloToPPHloOpConverter<mhlo::BitcastConvertOp>
    : public OpConversionPattern<mhlo::BitcastConvertOp> {
private:
  const ValueVisibilityMap &vis_;

public:
  HloToPPHloOpConverter(TypeConverter &typeConverter, MLIRContext *context,
                        const ValueVisibilityMap &vis)
      : OpConversionPattern<mhlo::BitcastConvertOp>(typeConverter, context),
        vis_(vis) {}

  LogicalResult
  matchAndRewrite(mhlo::BitcastConvertOp op,
                  mhlo::BitcastConvertOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result_vis = vis_.getValueVisibility(op.getResult());

    Type resultType = HloToPPHloTypeConverter::getTypeWithVisibility(
        this->getTypeConverter()->convertType(op.getType()), result_vis);

    auto in_type_size = op->getOperandTypes()[0]
                            .dyn_cast<RankedTensorType>()
                            .getElementTypeBitWidth();
    auto out_type_size = op->getResultTypes()[0]
                             .dyn_cast<RankedTensorType>()
                             .getElementTypeBitWidth();

    PPU_ENFORCE(in_type_size == out_type_size);

    rewriter.replaceOpWithNewOp<pphlo::BitcastConvertOp>(
        op, resultType, adaptor.getOperands()[0], out_type_size);

    return success();
  }
};

struct HloLegalizeToPPHlo
    : public HloLegalizeToPPHloPassBase<HloLegalizeToPPHlo> {
private:
  void populateHLOToPPHloConversionPattern(HloToPPHloTypeConverter &converter,
                                           OwningRewritePatternList &patterns,
                                           const ValueVisibilityMap &vis_map) {
    auto *context = patterns.getContext();

    patterns.insert<
        FuncOpConverter, ReturnOpConverter, HloCompToPPHloOpConverter,
        RSqrtOpConverter, ReduceOpConverter<mhlo::ReduceOp>,
        ReduceOpConverter<mhlo::ReduceWindowOp>, WhileOpConverter,
        IfOpConverter, HloToPPHloOpConverter<mhlo::AbsOp>,
        HloToPPHloOpConverter<mhlo::AddOp>, HloToPPHloOpConverter<mhlo::AndOp>,
        HloToPPHloOpConverter<mhlo::BitcastConvertOp>,
        HloToPPHloOpConverter<mhlo::BroadcastInDimOp>,
        HloToPPHloOpConverter<mhlo::CeilOp>,
        HloToPPHloOpConverter<mhlo::ClampOp>,
        HloToPPHloOpConverter<mhlo::ConcatenateOp>,
        HloToPPHloOpConverter<mhlo::ConstOp>,
        HloToPPHloOpConverter<mlir::ConstantOp>,
        HloToPPHloOpConverter<mhlo::ConvertOp>,
        HloToPPHloOpConverter<mhlo::ConvOp>, HloToPPHloOpConverter<mhlo::DivOp>,
        HloToPPHloOpConverter<mhlo::DotOp>, HloToPPHloOpConverter<mhlo::ExpOp>,
        HloToPPHloOpConverter<mhlo::FloorOp>,
        HloToPPHloOpConverter<mhlo::GatherOp>,
        HloToPPHloOpConverter<mhlo::IotaOp>, HloToPPHloOpConverter<mhlo::LogOp>,
        HloToPPHloOpConverter<mhlo::Log1pOp>,
        HloToPPHloOpConverter<mhlo::LogisticOp>,
        HloToPPHloOpConverter<mhlo::MaxOp>, HloToPPHloOpConverter<mhlo::MinOp>,
        HloToPPHloOpConverter<mhlo::MulOp>, HloToPPHloOpConverter<mhlo::NegOp>,
        HloToPPHloOpConverter<mhlo::NotOp>, HloToPPHloOpConverter<mhlo::OrOp>,
        HloToPPHloOpConverter<mhlo::PadOp>, HloToPPHloOpConverter<mhlo::PowOp>,
        HloToPPHloOpConverter<mhlo::ReshapeOp>,
        HloToPPHloOpConverter<mhlo::ReturnOp>,
        HloToPPHloOpConverter<mhlo::ReverseOp>,
        HloToPPHloOpConverter<mhlo::RngUniformOp>,
        HloToPPHloOpConverter<mhlo::SelectOp>,
        HloToPPHloOpConverter<mhlo::ShiftLeftOp>,
        HloToPPHloOpConverter<mhlo::SliceOp>,
        HloToPPHloOpConverter<mhlo::ShiftRightLogicalOp>,
        HloToPPHloOpConverter<mhlo::SortOp>,
        HloToPPHloOpConverter<mhlo::SqrtOp>, HloToPPHloOpConverter<mhlo::SubOp>,
        HloToPPHloOpConverter<mhlo::TransposeOp>,
        HloToPPHloOpConverter<mhlo::XorOp>>(converter, context, vis_map);
  }

  IoVisibilityInfo vis_info_;

  void parseVisibilityString() {
    if (io_visibility_json_.empty()) {
      return;
    }
    llvm::raw_os_ostream os(std::cout);
    if (auto json_v = llvm::json::parse(io_visibility_json_)) {
      llvm::json::Path::Root r;
      llvm::json::ObjectMapper map(*json_v, r);
      std::vector<std::string> str_vis;
      if (map && map.map("inputs", str_vis)) {
        vis_info_.convertFromStrings(str_vis);
      } else {
        r.printErrorContext(*json_v, os);
      }
    } else {
      handleAllErrors(json_v.takeError(), [&](const llvm::ErrorInfoBase &E) {
        os << "Failed to parse visibility JSON >>> " << io_visibility_json_
           << " <<<: " << E.message();
      });
    }
    os.flush();
  }

public:
  HloLegalizeToPPHlo(const HloLegalizeToPPHlo &) = default;
  HloLegalizeToPPHlo() = default;
  explicit HloLegalizeToPPHlo(const std::string &io_visibility_json) {
    io_visibility_json_ = io_visibility_json;
  }

  void runOnOperation() override {
    // This is a must step for cli workflow
    parseVisibilityString();

    auto &context = getContext();

    context.getTypeUniquer();

    OwningRewritePatternList patterns(&context);
    ConversionTarget target(context);
    HloToPPHloTypeConverter converter;

    // To pphlo dialect, ModuleOp is also a thing that we won't handle.
    target.addLegalDialect<PPHloDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    // After conversion, there shouldn't be any mhlo dialect thingy left.
    target.addIllegalDialect<mhlo::MhloDialect>();
    target.addIllegalDialect<scfhlo::SCFHLODialect>();

    // FcnOp is only legitimate iff signature and body is legal
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return converter.isSignatureLegal(op.getType()) &&
             converter.isLegal(&op.getBody());
    });
    // We keep mlir return op legal here.
    target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp op) {
      return converter.isLegal(op.getOperandTypes());
    });

    // Stage 1: Run a visibility discover pass to tag all Values' visibility
    ValueVisibilityMap vis_map = VisibilityDiscovery(getOperation(), vis_info_);

    // Stage 2: Do an actual dialect conversion.
    populateHLOToPPHloConversionPattern(converter, patterns, vis_map);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeToPPHloPass(const std::string &io_visibility_json) {
  return std::make_unique<HloLegalizeToPPHlo>(io_visibility_json);
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToPPHloPass() {
  return std::make_unique<HloLegalizeToPPHlo>();
}

} // namespace mlir::pphlo
