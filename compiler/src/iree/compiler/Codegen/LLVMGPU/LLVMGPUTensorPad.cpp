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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/SideEffectUtils.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"



#define DEBUG_TYPE "iree-llvmgpu-tensor-pad"

namespace mlir {
namespace iree_compiler {


  // static Operation * findRootOp() {}


namespace {
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
    // OpBuilder builder(&getContext());

    // RewritePatternSet paddingPattern(funcOp.getContext());

    // Find the linalg op (for now we'll just do one)
    linalg::GenericOp linalgOp;
    std::string anchorOpName;
    funcOp.walk(
        [&anchorOpName, &linalgOp](linalg::GenericOp op) {
        llvm::dbgs() << "I found a linalg generic \n" << op->getName().getStringRef().data() << "\n";
         StringRef str = op->getName().getStringRef();
         anchorOpName = std::string(str.begin(), str.end());
         linalgOp = op;
        });

    /// tensor.load ops to pad.
    llvm::MapVector<int64_t, Value> dispatchTensorLoads;

    // Find the tensor load ops feeding into the linalg
    for (OpOperand *opOperand : linalgOp.getInputAndOutputOperands()) {
      int64_t operandNumber = opOperand->getOperandNumber();
      Operation *op = opOperand->get().getDefiningOp();

      if (auto tensorLoad = dyn_cast_or_null<IREE::Flow::DispatchTensorLoadOp>(op)) {
        dispatchTensorLoads[operandNumber] = tensorLoad;
        llvm::dbgs() << "I've found a DispatchTensorLoadOp at position: (" << operandNumber << ")\n";
      }
    }

    // for (auto v : dispatchTensorLoads) {
    //   IREE::Flow::DispatchTensorLoadOp dispatchload =
    //       cast<IREE::Flow::DispatchTensorLoadOp>(v.second.getDefiningOp());
    // }


    // Setup builder
    OpBuilder builder(linalgOp);
    Location loc = linalgOp.getLoc();
    // Set the insertion point to before the linalg generic
    // builder.setInsertionPoint(linalgOp);

    // Operand 0 (input) needs its 1 dimension padded from 48 to 64
    IREE::Flow::DispatchTensorLoadOp dispatchLoad0 = cast<IREE::Flow::DispatchTensorLoadOp>(dispatchTensorLoads[0].getDefiningOp());
    Value paddingValue0 = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(dispatchTensorLoads[0].getType().cast<ShapedType>().getElementType()));
    SmallVector<int64_t> paddedShape = {32, 32};
    auto paddedTensorType0 = RankedTensorType::get(
      paddedShape, getElementTypeOrSelf(dispatchLoad0));
    Value paddedInputValue = linalg::makeComposedPadHighOp(builder, loc, paddedTensorType0, dispatchTensorLoads[0], paddingValue0, false);

    // Operand 1 (output) needs its 0 dimension padded from 48 to 64
    IREE::Flow::DispatchTensorLoadOp dispatchLoad1 = cast<IREE::Flow::DispatchTensorLoadOp>(dispatchTensorLoads[1].getDefiningOp());
    Value paddingValue1 = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(dispatchTensorLoads[1].getType().cast<ShapedType>().getElementType()));
    auto paddedTensorType1 = RankedTensorType::get(
      paddedShape, getElementTypeOrSelf(dispatchLoad1));
    Value paddedOutputValue = linalg::makeComposedPadHighOp(builder, loc, paddedTensorType1, dispatchTensorLoads[1], paddingValue1, false);

    // Replace the input into the linalg with the padded values.
    if (false) {
    linalgOp.getInputOperand(0)->set(paddedInputValue);
    linalgOp.getOutputOperand(0)->set(paddedOutputValue);
    }
    // This doesn't work, setting inputs needs to match the op in question

    llvm::dbgs() << "Done with runOnOperation()\n";
  }

};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTensorPadPass() {
  return std::make_unique<LLVMGPUTensorPadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
