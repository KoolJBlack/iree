// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"


namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static Operation *sliceTensor(Location loc, Value expanded, Value original,
                              OpBuilder &builder) {
  auto originalType = original.getType().cast<RankedTensorType>();
  auto rank = originalType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getI64IntegerAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  SmallVector<OpFoldResult> sizes(rank);
  for (int i = 0, e = rank; i < e; ++i) {
    if (!originalType.isDynamicDim(i)) {
      sizes[i] = builder.getI64IntegerAttr(originalType.getDimSize(i));
    } else {
      sizes[i] = builder.create<tensor::DimOp>(loc, original, i).getResult();
    }
  }

  return builder.create<tensor::ExtractSliceOp>(loc, expanded, offsets, sizes,
                                                strides);
}

static bool isSimpleTranspose(linalg::GenericOp op) {
  if (!op) return false;
  if (op.getNumDpsInputs() != 1) return false;
  if (op.getNumDpsInits() != 1) return false;
  if (!op.hasTensorSemantics()) return false;
  if (op.getNumReductionLoops() > 0) return false;
  auto inputOperand = op.getDpsInputOperand(0);
  auto inputIndexMap = op.getMatchingIndexingMap(inputOperand);
  if (!inputIndexMap.isPermutation() || inputIndexMap.isIdentity())
    return false;
  auto outputOperand = op.getDpsInitOperand(0);
  auto outputIndexingMap = op.getMatchingIndexingMap(outputOperand);
  if (!outputIndexingMap.isIdentity()) return false;
  return true;
}

static bool padTensor(Location loc, OpOperand *operand,
                      llvm::ArrayRef<int64_t> alignments, OpBuilder &builder) {
  Value original = operand->get();
  auto type = original.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = type.getShape();

  // New dimensions.
  SmallVector<int64_t> newStaticDims(shape.begin(), shape.end());
  SmallVector<OpFoldResult> newPaddingSizes(shape.size(),
                                            builder.getI64IntegerAttr(0));

  // Compute padded dims.
  bool needsPad = false;

  llvm::dbgs() << "PadTransposeOp: padding to new static dim: \n";
  for (int i = 0, e = shape.size(); i < e; ++i) {
    auto inputDim = shape[i];
    auto alignment = alignments[i];
    assert(inputDim >= 0);
    // Static dim.
    if ((inputDim % alignment) == 0) {
      llvm::dbgs() << "already aligned dim at i(" << i << ") from " << inputDim <<  " with alignment: " << alignment<< " \n";

      newStaticDims[i] = inputDim;
      continue;
    }
    int64_t alignedDim = (inputDim + (alignment - 1)) & ~(alignment - 1);

    llvm::dbgs() << "dim at i(" << i << ") from " << inputDim <<  " to "  <<  alignedDim << " with alignment: " << alignment<< " \n";

    newStaticDims[i] = alignedDim;
    newPaddingSizes[i] = builder.getI64IntegerAttr(alignedDim - inputDim);
    needsPad = true;
  }
  if (!needsPad){
    return false;
  }

  auto resultType = RankedTensorType::get(newStaticDims, type.getElementType());
  Value zeroConstant = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(type.getElementType()));
  SmallVector<OpFoldResult> zeroStaticLow(shape.size(),
                                          builder.getI64IntegerAttr(0));
  SmallVector<Value> nullLow;
  Value padded = builder.create<tensor::PadOp>(loc, resultType, operand->get(),
                                               zeroStaticLow, newPaddingSizes,
                                               zeroConstant);
  operand->set(padded);
  return true;
}

static FailureOr<Value> newPadTensor(PatternRewriter &rewriter, Location loc, OpOperand *operand,
                      llvm::ArrayRef<int64_t> alignments ) {
  Operation* op = operand->get().getDefiningOp();
  //   if (!tensorLoad) {
  //   return rewriter.notifyMatchFailure(linalgOp, "does not have tensor load");
  // }

  Value originalValue = operand->get();
  auto originalType = originalValue.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> originalShape = originalType.getShape();


  // Determine the padded shape using the alignment
  SmallVector<int64_t> paddedShape(originalShape.begin(), originalShape.end());
  llvm::dbgs() << "PadTransposeOp: padding to new static dim: \n";
  bool needsPad = false;
  for (int i = 0, e = originalShape.size(); i < e; ++i) {
    auto inputDim = originalShape[i];
    auto alignment = alignments[i];
    assert(inputDim >= 0);
    // Static dim.
    if ((inputDim % alignment) == 0) {
      paddedShape[i] = inputDim;
      continue;
    }
    int64_t alignedDim = (inputDim + (alignment - 1)) & ~(alignment - 1);

    llvm::dbgs() << "from " << inputDim <<  " to "  <<  alignedDim << " \n";

    paddedShape[i] = alignedDim;
    needsPad = true;
  }
  if (!needsPad) {
    return rewriter.notifyMatchFailure(op, "Op does not need to be padded");
  }

  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(originalType.getElementType()));
  auto paddedTensorResultType =
      RankedTensorType::get(paddedShape, originalType.getElementType());

  auto zero = rewriter.getI64IntegerAttr(0);
  SmallVector<OpFoldResult> low(paddedTensorResultType.getRank(), zero);
  SmallVector<OpFoldResult> high(paddedTensorResultType.getRank(), zero);
  Value paddedValue = rewriter.create<tensor::PadOp>(loc, paddedTensorResultType, operand->get(), low, high, paddingValue, /*nofold=*/false);

  return paddedValue;
}

// Pads the leading (fastest moving) dimension of the operand to a multiple of the alignment size
static bool padLeadingDim(PatternRewriter &rewriter, Location loc,
                          linalg::LinalgOp linalgOp, OpOperand *opOperand,
                          int64_t alignment) {
  Value originalValue = opOperand->get();
  auto originalType = originalValue.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> originalShape = originalType.getShape();

  // All alignments are 1 except for leading dimension.
  SmallVector<int64_t> alignments(originalShape.size(), 1);
  alignments.back() = alignment;


  // // Determine the padded shape from the load
  // ArrayRef<int64_t> shape = linalgOp.getShape(opOperand);
  // SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
  // for (const auto &[index, size] :
  //       llvm::enumerate(tensorLoad.getMixedSizes())) {
  //   if (Optional<int64_t> cst = getConstantIntValue(size)) {
  //     paddedShape[index] = cst.value();
  //   } else {
  //     FailureOr<int64_t> upperBound =
  //         linalg::getConstantUpperBoundForIndex(size.get<Value>());
  //     if (failed(upperBound)) {
  //       return rewriter.notifyMatchFailure(
  //           linalgOp, "No constant bounding box can be found for padding");
  //     }
  //     paddedShape[index] = *upperBound;
  //   }
  // }

  // Now pad
  padTensor(loc, opOperand, alignments, rewriter);

  return true;
}

static FailureOr<SmallVector<Value>> rewriteAsPaddedOp(
    IRRewriter &rewriter, linalg::LinalgOp linalgOp,
    linalg::LinalgOp &paddedOp) {
  Location loc = linalgOp.getLoc();

  IRRewriter::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(linalgOp);

  // Pad each input operand in shared memory up to the targets bounding box
  // size. In this case, this corresponds with the maximum tile size from
  // distributing to workgroups.
  SmallVector<Value> paddedOperands;
  paddedOperands.reserve(linalgOp.getNumDpsInputs() +
                         linalgOp.getNumDpsInits());
  for (OpOperand &opOperand : linalgOp->getOpOperands()) {
    // Find DispatchTensorLoadOp's feeding into the linalg or abort.
    auto tensorLoad = dyn_cast_or_null<IREE::Flow::DispatchTensorLoadOp>(
        opOperand.get().getDefiningOp());
    if (!tensorLoad) {
      return rewriter.notifyMatchFailure(linalgOp, "does not have tensor load");
    }

    // Determine the padded shape from the load
    ArrayRef<int64_t> shape = linalgOp.getShape(&opOperand);
    SmallVector<int64_t> paddedShape(shape.begin(), shape.end());
    for (const auto &[index, size] :
         llvm::enumerate(tensorLoad.getMixedSizes())) {
      if (Optional<int64_t> cst = getConstantIntValue(size)) {
        paddedShape[index] = cst.value();
      } else {
        FailureOr<int64_t> upperBound =
            linalg::getConstantUpperBoundForIndex(size.get<Value>());
        if (failed(upperBound)) {
          return rewriter.notifyMatchFailure(
              linalgOp, "No constant bounding box can be found for padding");
        }
        paddedShape[index] = *upperBound;
      }
    }

    Value paddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(getElementTypeOrSelf(tensorLoad)));
    auto paddedTensorType =
        RankedTensorType::get(paddedShape, getElementTypeOrSelf(tensorLoad));
    Value paddedValue = linalg::makeComposedPadHighOp(
        rewriter, loc, paddedTensorType, tensorLoad, paddingValue,
        /*nofold=*/false);
    paddedOperands.push_back(paddedValue);
  }

  // Clone linalgOp to paddedOp with padded input/output shapes.
  auto resultTensorTypes = ValueRange(paddedOperands)
                               .take_back(linalgOp.getNumDpsInits())
                               .getTypes();
  paddedOp = mlir::clone(rewriter, linalgOp, resultTensorTypes, paddedOperands);

  // Slice out the original shape from the padded result to pass on to
  // consumers. The original linalg op is used to provide the dims for the reify
  // result shapes.
  SmallVector<SmallVector<Value>> reifiedResultShapes;
  if (failed(cast<ReifyRankedShapedTypeOpInterface>(linalgOp.getOperation())
                 .reifyResultShapes(rewriter, reifiedResultShapes))) {
    return failure();
  }

  SmallVector<Value> paddedSubviewResults;
  paddedSubviewResults.reserve(paddedOp->getNumResults());
  for (const auto &[resultNumber, paddedResult] :
       llvm::enumerate(paddedOp->getResults())) {
    int64_t rank = paddedResult.getType().cast<RankedTensorType>().getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (Value v : reifiedResultShapes[resultNumber])
      sizes.push_back(getAsOpFoldResult(v));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubviewResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, sizes, strides));
  }
  return paddedSubviewResults;
}

namespace {

/// A pattern to pad linalg to the lowest dimension.
// A pattern to single op linalg.generic transposes to the lowest dimension.
class PadTransposeOp : public OpInterfaceRewritePattern<linalg::LinalgOp> {
 public:
  PadTransposeOp(MLIRContext *context, int size, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), paddingSize(size) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    Location loc = linalgOp.getLoc();

    // Single op in/out.
    if (linalgOp.getNumDpsInits() != 1) return failure();
    if (linalgOp.getNumDpsInputs() != 1) return failure();

    // Checks preconditions for transpose.
    LinalgOpInfo opInfo(linalgOp);
    if (!opInfo.isTranspose() || opInfo.isDynamic() || opInfo.isReduction() ||
        !isa<linalg::GenericOp>(linalgOp)) {
      return rewriter.notifyMatchFailure(linalgOp, "transpose preconditions");
    }

    // Check if inputs have a shaped type and padding is needed.
    bool isPaddingNeeded = false;
    OpOperand *opOperand = linalgOp.getDpsInputOperand(0);
    auto tensorType = opOperand->get().getType().dyn_cast<RankedTensorType>();
    if (!tensorType || !tensorType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(linalgOp, "op operands aren't tensor type");
    }

    //TODO: this is a hack. For now only look at 2d transposes.
    if (tensorType.getShape().size() != 2) {
      return failure();
    }

    for (auto dimIndexSize : llvm::enumerate(tensorType.getShape())) {
      // if (dimIndexSize.index() == 0 && dimIndexSize.value() == 1) {
      //   // We dont' care about leading batch dimensions that are 1
      //   continue;
      // }
      if (!isPaddingNeeded && dimIndexSize.value() % paddingSize != 0)
        isPaddingNeeded = true;
    }
    if (!isPaddingNeeded) return failure();

    llvm::dbgs() << "PadTransposeOp: Is this our starting transpose? \n";
    linalgOp.dump();
    llvm::dbgs() << opInfo.isTranspose() << "\n\n";



    // create a new operand
    opOperand->get().dump();
    llvm::dbgs() <<"***\n";
    padLeadingDim(rewriter, loc, linalgOp, opOperand, paddingSize);

    OpOperand *output = linalgOp.getDpsInitOperand(0);
    Value origOutput = output->get();
    OpResult result = linalgOp.getOperation()->getResult(0);
    if (padLeadingDim(rewriter, loc,linalgOp, output, paddingSize)) {
      result.setType(output->get().getType());

      rewriter.setInsertionPoint(linalgOp.getOperation());
      Operation *slicedResult = sliceTensor(loc, result, origOutput, rewriter);
      result.replaceAllUsesWith(slicedResult->getResult(0));
    }

    return success();
  }

 private:
  int paddingSize;
};

/// A pattern to switch transpose and pad with pad and transpose when the
/// tranpose output has an unaligned leading dimension.
struct TransposePadToPadTransposeOp : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  TransposePadToPadTransposeOp(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!padOp.hasZeroLowPad())
      return rewriter.notifyMatchFailure(padOp, "expected zero low pad");

    auto genericOpTranspose = padOp.getSource().getDefiningOp<linalg::GenericOp>();
    if (!isSimpleTranspose(genericOpTranspose)) {
      return rewriter.notifyMatchFailure(
          padOp, "expected source to be a simple transpose op");
    }

    // Create a new PadOp.
    llvm::dbgs() << "TransposePadToPadTransposeOp: This is our starting pad: \n";
    // Setup: We have a transpose, then pad (both printed) followed by matmul (not printed)
    genericOpTranspose.dump();
    padOp.dump();
    llvm::dbgs() << "\n";

    // Apply reverse transpose to get the low/high paddings and the new shape.
    OpOperand *transposeInput = genericOpTranspose.getDpsInputOperand(0);
    AffineMap indexingMap = genericOpTranspose.getMatchingIndexingMap(transposeInput);

    // The old pad is after the transpose feeding into matmul
    auto oldHiPad = padOp.getMixedHighPad();
    SmallVector<OpFoldResult> newHiPad(oldHiPad);
    RankedTensorType oldPadType = padOp.getResultType();
    ArrayRef<int64_t> oldPadShape = oldPadType.getShape();
    SmallVector<int64_t> newShape(oldPadShape);

    for (auto en : enumerate(indexingMap.getResults())) {
      unsigned pos = en.value().cast<AffineDimExpr>().getPosition();
      unsigned index = en.index();
      newHiPad[pos] = oldHiPad[index];
      newShape[pos] = oldPadShape[index];
    }
    auto newPadResultType =
        RankedTensorType::get(newShape, oldPadType.getElementType());
    // This is the new pad that will go before the transpose, not
    auto newPadOp = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), newPadResultType, transposeInput->get(),
        padOp.getMixedLowPad(), newHiPad, padOp.getConstantPaddingValue());

    // Reuse the old PadOp for the init (output) operand for the transpose???
    // Originally, this is a Tensor.empty(). Although he pads this out to be correct,
    /// it might go away in subsequent optimization.
    auto newPadOpForInit = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), padOp.getResultType(),
        genericOpTranspose.getDpsInitOperand(0)->get(), padOp.getMixedLowPad(),
        padOp.getMixedHighPad(), padOp.getConstantPaddingValue());

    newPadOpForInit.setOperand(0, genericOpTranspose.getDpsInitOperand(0)->get());

    auto newTranspose = rewriter.create<linalg::GenericOp>(
        padOp.getLoc(), padOp.getResultType(), newPadOp->getResult(0),
        newPadOpForInit->getResult(0), genericOpTranspose.getIndexingMapsArray(),
        genericOpTranspose.getIteratorTypesArray(),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOpTranspose));
    rewriter.inlineRegionBefore(genericOpTranspose.getRegion(), newTranspose.getRegion(),
                                newTranspose.getRegion().begin());
    rewriter.replaceOp(padOp, newTranspose->getResult(0));

    llvm::dbgs() << "Results after pad change: \n";
    newPadOp.dump();
    newPadOpForInit.dump();
    newTranspose.dump();
    llvm::dbgs() << "---\n";
    llvm::dbgs() << "Input into new pad: \n";
    auto globalLoadOp = newPadOp.getSource().getDefiningOp<IREE::Util::GlobalLoadOp>();
    if (globalLoadOp) {
      globalLoadOp.dump();
      llvm::dbgs() << "getGlobal: " << globalLoadOp.getGlobal() << " getGlobalAttrName: "<<  globalLoadOp.getGlobalAttrName() <<  "\n";
      globalLoadOp.getType().dump();
      llvm::dbgs() << "\n---\n";

      // auto globalLoadedValue = globalLoadOp.;
      // auto globalOpName = globalLoadOp.getGlobal();
      llvm::dbgs() << "\n---\n";

    }


    return success();
  }
};

/// A pattern to pad statically shaped matmul operands to the next integer
/// multiple of padSize.
class PadMatmulOp : public OpInterfaceRewritePattern<linalg::LinalgOp> {
 public:
  PadMatmulOp(MLIRContext *context, int size, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), paddingSize(size) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = linalgOp.getOperation();
    const bool isBatchMatmul = isa<linalg::BatchMatmulOp>(op);
    const bool isMatmul = isa<linalg::MatmulOp>(op);
    if (!isBatchMatmul && !isMatmul) return failure();

    // llvm::dbgs() << "PadMatmulOp: starting matmul \n";
    // linalgOp.dump();

    Location loc = linalgOp.getLoc();
    Value lhs = linalgOp.getDpsInputOperand(0)->get();
    Value rhs = linalgOp.getDpsInputOperand(1)->get();
    Value result = linalgOp.getDpsInitOperand(0)->get();

    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    auto resultType = result.getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType) return failure();

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return failure();

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    const int B = isBatchMatmul ? lhsShape[0] : -1;
    const int M = isBatchMatmul ? lhsShape[1] : lhsShape[0];
    const int K = lhsShape.back(), N = rhsShape.back();

    int newMSize = std::ceil(float(M) / paddingSize) * paddingSize;
    int newNSize = std::ceil(float(N) / paddingSize) * paddingSize;
    int newKSize = std::ceil(float(K) / paddingSize) * paddingSize;

    int paddingForM = newMSize - M;
    int paddingForN = newNSize - N;
    int paddingForK = newKSize - K;

    if (paddingForM == 0 && paddingForN == 0 && paddingForK == 0)
      return failure();

    auto getFullShape = [&](ArrayRef<int> dims) {
      SmallVector<int64_t, 3> shape;
      if (isBatchMatmul) shape.push_back(B);
      llvm::append_range(shape, dims);
      return shape;
    };

    auto lhsPaddedType = RankedTensorType::get(
        getFullShape({newMSize, newKSize}), lhsType.getElementType());

    auto rhsPaddedType = RankedTensorType::get(
        getFullShape({newKSize, newNSize}), rhsType.getElementType());

    Value lhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(lhsType.getElementType()));

    Value rhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(rhsType.getElementType()));

    auto createPadding = [&](ArrayRef<int64_t> padding) {
      SmallVector<OpFoldResult> result;
      if (isBatchMatmul) {
        result.push_back(rewriter.getI64IntegerAttr(0));
      }
      for (auto pad : padding) {
        result.push_back(rewriter.getI64IntegerAttr(pad));
      }
      return result;
    };

    Value paddedLhs = lhs;
    if (paddingForM > 0 || paddingForK > 0) {
      paddedLhs = rewriter.create<tensor::PadOp>(
          loc, lhsPaddedType, lhs, createPadding({0, 0}),
          createPadding({paddingForM, paddingForK}), lhsPaddingValue);
    }

    Value paddedRhs = rhs;
    if (paddingForK > 0 || paddingForN > 0) {
      paddedRhs = rewriter.create<tensor::PadOp>(
          loc, rhsPaddedType, rhs, createPadding({0, 0}),
          createPadding({paddingForK, paddingForN}), rhsPaddingValue);
    }

    // Padding for K-dim doesn't change result size.
    if (paddingForM == 0 && paddingForN == 0) {
      auto paddedMatmulOp =
          mlir::clone(rewriter, linalgOp, {resultType},
                      ArrayRef<Value>{paddedLhs, paddedRhs, result});

      llvm::dbgs() << "PadMatmulOp: starting matmul \n";
      linalgOp.dump();
      llvm::dbgs() << "Final padded mamtul K: \n";
      paddedLhs.dump();
      paddedRhs.dump();
      paddedMatmulOp->dump();
      llvm::dbgs() << "---\n";

      rewriter.replaceOp(linalgOp, paddedMatmulOp->getResults());
    } else {
      auto newResultType = RankedTensorType::get(
          getFullShape({newMSize, newNSize}), resultType.getElementType());
      Value resultPaddingValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(resultType.getElementType()));
      Value paddedResult = rewriter.create<tensor::PadOp>(
          loc, newResultType, result, createPadding({0, 0}),
          createPadding({paddingForM, paddingForN}), resultPaddingValue);
      auto paddedMatmulOp =
          mlir::clone(rewriter, linalgOp, {newResultType},
                      ArrayRef<Value>{paddedLhs, paddedRhs, paddedResult});

      llvm::dbgs() << "PadMatmulOp: starting matmul \n";
      linalgOp.dump();
      llvm::dbgs() << "Final padded mamtul: \n";
      paddedLhs.dump();
      paddedRhs.dump();
      paddedMatmulOp->dump();
      llvm::dbgs() << "---\n";

      auto zero = rewriter.getI64IntegerAttr(0);
      auto one = rewriter.getI64IntegerAttr(1);
      auto mAttr = rewriter.getIndexAttr(M);
      auto nAttr = rewriter.getIndexAttr(N);
      SmallVector<OpFoldResult> offsets, strides, sizes;
      if (isBatchMatmul) {
        offsets.assign(3, zero);
        strides.assign(3, one);
        sizes = {rewriter.getIndexAttr(B), mAttr, nAttr};
      } else {
        offsets.assign(2, zero);
        strides.assign(2, one);
        sizes = {mAttr, nAttr};
      }
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          linalgOp, paddedMatmulOp->getResults()[0], offsets, sizes, strides);
    }

    return success();
  }

 private:
  int paddingSize;
};

class PadLinalgOpsPass : public PadLinalgOpsBase<PadLinalgOpsPass> {
 public:
  PadLinalgOpsPass(int size) : paddingSize(size) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    llvm::dbgs() << "runOnOperation() -- paddingSize: " << paddingSize <<"\n";
    MLIRContext *context = &getContext();
    auto moduleOp = getOperation();
    llvm::dbgs() << "ModuleOP name: " << moduleOp->getName() << "\n";

    SymbolTable symbolTable(moduleOp);

    RewritePatternSet patterns(context);
    patterns.insert<PadMatmulOp>(context, paddingSize);
    llvm::dbgs() << "runOnOperation() -- applying PadMatmulOp pattern\n";
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // patterns.clear();
    // patterns.insert<TransposePadToPadTransposeOp>(context);
    // llvm::dbgs() << "runOnOperation() -- applying TransposePadToPadTransposeOp pattern\n";
    // if (failed(applyPatternsAndFoldGreedily(getOperation(),
    //                                         std::move(patterns)))) {
    //   return signalPassFailure();
    // }

    patterns.clear();
    patterns.insert<PadTransposeOp>(context, paddingSize);
    llvm::dbgs() << "runOnOperation() -- applying PadTransposeOp pattern\n";
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  int paddingSize;
};

}  // namespace

std::unique_ptr<Pass> createPadLinalgOpsToIntegerMultiplePass(int paddingSize) {
  // llvm::dbgs() << "createPadLinalgOpsToIntegerMultiplePass() -- paddingSize: " << paddingSize <<"\n";
  return std::make_unique<PadLinalgOpsPass>(paddingSize);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
