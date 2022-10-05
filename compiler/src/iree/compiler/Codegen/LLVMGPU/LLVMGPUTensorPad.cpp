// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"
#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/LLVMGPU/TransposeUtils.h"


#define DEBUG_TYPE "iree-llvmgpu-tensor-pad"

namespace mlir {
namespace iree_compiler {

namespace {



static FailureOr<SmallVector<Value>> rewriteAsPaddedOp(
    PatternRewriter &builder, linalg::LinalgOp linalgOp, linalg::LinalgOp &paddedOp) {
  Location loc = linalgOp.getLoc();

  OpBuilder::InsertionGuard g(builder);
  // Set IP after op because we also take the dims of the original output.
  builder.setInsertionPointAfter(linalgOp);

  // Pad each operand out to 32,32
  SmallVector<int64_t> paddedShape = {32, 32};
  SmallVector<Value> paddedOperands;
  paddedOperands.reserve(linalgOp.getNumInputsAndOutputs());
  for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
    // Find DispatchTensorLoadOp's feeding into the linalg or abort.
    auto tensorLoad = dyn_cast_or_null<IREE::Flow::DispatchTensorLoadOp>(
        opOperand->get().getDefiningOp());
    if (!tensorLoad) {
      return builder.notifyMatchFailure(linalgOp, "does not have tensor load");
    }

    Value paddingValue = builder.create<arith::ConstantOp>(
        loc, builder.getZeroAttr(
                 tensorLoad.getType().cast<ShapedType>().getElementType()));
    auto paddedTensorType0 =
        RankedTensorType::get(paddedShape, getElementTypeOrSelf(tensorLoad));
    Value paddedValue = linalg::makeComposedPadHighOp(
        builder, loc, paddedTensorType0, tensorLoad, paddingValue,
        false);
    paddedOperands.push_back(paddedValue);
  }

  // Clone linalgOp to paddedOp with padded input/output shapes.
  auto resultTensorTypes =
      ValueRange(paddedOperands).take_back(linalgOp.getNumOutputs()).getTypes();
  paddedOp = linalgOp.clone(builder, loc, resultTensorTypes, paddedOperands);

  // Slice out the original shape from the padded result to pass on to
  // consumers. The original linalg op is used to provide the dims for the reify
  // result shapes.
  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(cast<ReifyRankedShapedTypeOpInterface>(linalgOp.getOperation())
                 .reifyResultShapes(builder, reifiedResultShapes))) {
    return failure();
  }

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

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
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
    LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);
    // Checks preconditions for shared mem transpose. Only pad if op is dynamic.
    if (!opInfo.isTranspose() || !opInfo.isDynamic()) {
      return rewriter.notifyMatchFailure(linalgOp, "failed preconditions");
    }

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
        &getContext(),
        linalg::LinalgTransformationFilter(
            ArrayRef<StringAttr>{},
            StringAttr::get(&getContext(), getTransposePadMarker())));
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // Remove all the markers at the end.
    funcOp->walk([&](linalg::LinalgOp op) {
      op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
    });

    llvm::dbgs() << "Done with runOnOperation()\n";
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass() {
  return std::make_unique<LLVMGPUTensorPadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
