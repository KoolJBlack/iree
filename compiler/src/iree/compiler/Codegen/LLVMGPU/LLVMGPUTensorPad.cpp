// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/LLVMGPU/TilingUtils.h"
#include "iree/compiler/Codegen/LLVMGPU/TransposeUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"

#define DEBUG_TYPE "iree-llvmgpu-tensor-pad"

namespace mlir {
namespace iree_compiler {

namespace {

static FailureOr<SmallVector<Value>> rewriteAsPaddedOp(
    OpBuilder &builder, linalg::LinalgOp linalgOp, linalg::LinalgOp &paddedOp) {
  Location loc = linalgOp.getLoc();

  /// tensor.load ops to pad.
  llvm::MapVector<int64_t, Value> dispatchTensorLoads;
  dispatchTensorLoads.reserve(linalgOp.getNumInputsAndOutputs());

  // Find the tensor load ops feeding into the linalg
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    int64_t operandNumber = opOperand->getOperandNumber();
    Operation *op = opOperand->get().getDefiningOp();

    if (auto tensorLoad =
            dyn_cast_or_null<IREE::Flow::DispatchTensorLoadOp>(op)) {
      dispatchTensorLoads[operandNumber] = tensorLoad;
      llvm::dbgs() << "I've found a DispatchTensorLoadOp at position: ("
                   << operandNumber << ")\n";
    }
  }

  OpBuilder::InsertionGuard g(builder);
  // Set IP after op because we also take the dims of the original output.
  builder.setInsertionPointAfter(linalgOp);

  // Operand 0 (input) needs its 1 dimension padded from 48 to 64
  IREE::Flow::DispatchTensorLoadOp dispatchLoad0 =
      cast<IREE::Flow::DispatchTensorLoadOp>(
          dispatchTensorLoads[0].getDefiningOp());
  Value paddingValue0 = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(dispatchTensorLoads[0]
                                   .getType()
                                   .cast<ShapedType>()
                                   .getElementType()));
  SmallVector<int64_t> paddedShape = {32, 32};
  auto paddedTensorType0 =
      RankedTensorType::get(paddedShape, getElementTypeOrSelf(dispatchLoad0));
  Value paddedInputValue = linalg::makeComposedPadHighOp(
      builder, loc, paddedTensorType0, dispatchTensorLoads[0], paddingValue0,
      false);

  // Operand 1 (output) needs its 0 dimension padded from 48 to 64
  IREE::Flow::DispatchTensorLoadOp dispatchLoad1 =
      cast<IREE::Flow::DispatchTensorLoadOp>(
          dispatchTensorLoads[1].getDefiningOp());
  Value paddingValue1 = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(dispatchTensorLoads[1]
                                   .getType()
                                   .cast<ShapedType>()
                                   .getElementType()));
  auto paddedTensorType1 =
      RankedTensorType::get(paddedShape, getElementTypeOrSelf(dispatchLoad1));
  Value paddedOutputValue = linalg::makeComposedPadHighOp(
      builder, loc, paddedTensorType1, dispatchTensorLoads[1], paddingValue1,
      false);

  // Clone `opToPad` to paddedOp with padded input/output shapes.
  SmallVector<Value> newOperands{paddedInputValue, paddedOutputValue};
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(linalgOp.getNumOutputs()).getTypes();
  paddedOp = linalgOp.clone(builder, loc, resultTensorTypes, newOperands);

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  // The reify result shapes will grab the dimensions of each output shape using
  // the source linalg op.
  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(cast<ReifyRankedShapedTypeOpInterface>(linalgOp.getOperation())
                 .reifyResultShapes(builder, reifiedResultShapes))) {
    return failure();
  }
  // TODO: don't think I need this assert?
  // assert(reifiedResultShapes.size() == opToPad->getNumResults() &&
  //        "expected same number of results");

  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(paddedOp->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = paddedResult.getType().cast<RankedTensorType>().getRank();
    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (Value v : reifiedResultShapes[resultNumber])
      sizes.push_back(getAsOpFoldResult(v));
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    paddedSubviewResults.push_back(builder.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, sizes, strides));
  }
  return paddedSubviewResults;
}

struct TransposePadOpPattern : public OpRewritePattern<linalg::GenericOp> {
 public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  TransposePadOpPattern(MLIRContext *context,
                        linalg::LinalgTransformationFilter filt)
      : OpRewritePattern<linalg::GenericOp>(context), filter(std::move(filt)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, linalgOp))) {
      return rewriter.notifyMatchFailure(linalgOp, "filter check");
    }

    // TODO: precondition checks
    linalg::LinalgOp paddedOp;
    FailureOr<SmallVector<Value>> newResults =
        rewriteAsPaddedOp(rewriter, linalgOp, paddedOp);
    if (failed(newResults)) {
      return failure();
    }

    // Replace the original operation to pad.
    rewriter.replaceOp(linalgOp, *newResults);
    filter.replaceLinalgTransformationFilter(
        rewriter, paddedOp);  // Note filter applied to replacement.

    return success();
  }

 private:
  mlir::linalg::LinalgTransformationFilter filter;
};

struct LLVMGPUTensorPadPass
    : public LLVMGPUTensorPadBase<LLVMGPUTensorPadPass> {
 private:
 public:
  LLVMGPUTensorPadPass() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override {
    llvm::dbgs() << "runOnOperation()\n";
    auto funcOp = getOperation();

    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<TransposePadOpPattern>(
        &getContext(), linalg::LinalgTransformationFilter(
                           ArrayRef<StringAttr>{},
                           StringAttr::get(&getContext(), "TRANSPOSE_PAD")));

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    llvm::dbgs() << "Done with runOnOperation()\n";
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass() {
  return std::make_unique<LLVMGPUTensorPadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
