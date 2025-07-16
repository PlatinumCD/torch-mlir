//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToAnalog/TorchToAnalog.h"

// TODO: include LLVMIR dialect

#include "../PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include <iostream>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// -----------------------------------------------------------------------------
// The pass
// -----------------------------------------------------------------------------
// Patterns for individual ops should live in one of the other files, and
// added via the relevant `populate*PatternsAndLegality` functions.
// This file is just for the pass definition itself.

//  ───  Conversion pattern  ────────────────────────────────────────────
struct MmToCall : OpConversionPattern<AtenMmOp> {
  using OpConversionPattern::OpConversionPattern;   // gives us TypeConverter

  LogicalResult matchAndRewrite(
      AtenMmOp mm, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = mm.getLoc();

    // Converted operands (RankedTensorType) are already provided by `adaptor`.
    Value lhs = adaptor.getSelf();        // tensor<3x4xf32>
    Value rhs = adaptor.getMat2();        // tensor<4x4xf32>

    // ------------------------------------------------------------------
    // 1. Result tensor prototype (converted from !torch.vtensor)
    // ------------------------------------------------------------------
    Type newResTy = getTypeConverter()->convertType(mm.getType());
    auto rankedTy = newResTy.cast<RankedTensorType>();
    Type elemTy   = rankedTy.getElementType();

    // Create `tensor.empty` with same (static) shape 3×4.
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, rankedTy.getShape(), elemTy);

    // Optional zero-fill so CUSTOM_OP sees a clean buffer.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elemTy, 0.0));
    Value out  =
        rewriter.create<linalg::FillOp>(loc, zero, init).getResult(0);

    // ------------------------------------------------------------------
    // 2. Ensure symbol @CUSTOM_OP exists with *converted* memref types
    // ------------------------------------------------------------------
    auto tensorToMemRef = [&](Value v) {
      auto t = v.getType().cast<RankedTensorType>();
      return MemRefType::get(t.getShape(), elemTy);         // identity layout
    };
    MemRefType outBufTy = tensorToMemRef(out);   // M × N
    MemRefType lhsBufTy = tensorToMemRef(lhs);   // M × K
    MemRefType rhsBufTy = tensorToMemRef(rhs);

    ModuleOp mod = mm->getParentOfType<ModuleOp>();
    FunctionType fnTy =
        rewriter.getFunctionType({outBufTy, lhsBufTy, rhsBufTy}, {});

    if (auto old = mod.lookupSymbol<func::FuncOp>("CUSTOM_OP"))
      if (old.getFunctionType() != fnTy) old.erase();

    if (!mod.lookupSymbol<func::FuncOp>("CUSTOM_OP")) {
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      rewriter.create<func::FuncOp>(loc, "CUSTOM_OP", fnTy).setPrivate();
    }

    // ------------------------------------------------------------------
    // 3. Cast tensors → memref, call, replace
    // ------------------------------------------------------------------
    auto toMemref = [&](Value t, MemRefType ty) {
      return rewriter.create<bufferization::ToMemrefOp>(loc, ty, t);
    };
    Value outBuf = toMemref(out, outBufTy);
    Value lhsBuf = toMemref(lhs, lhsBufTy);
    Value rhsBuf = toMemref(rhs, rhsBufTy);

    rewriter.create<func::CallOp>(loc, "CUSTOM_OP", TypeRange{},
                                  ValueRange{outBuf, lhsBuf, rhsBuf});
//                                  ValueRange{lhsBuf, rhsBuf, outBuf});

    // Cast the (now-filled) buffer back to the precise tensor result type.
    Value resultTensor =
        rewriter.create<bufferization::ToTensorOp>(loc, outBuf);
    resultTensor.getDefiningOp()->setAttr("restrict", rewriter.getUnitAttr());

    resultTensor =
        rewriter.create<tensor::CastOp>(loc, newResTy, resultTensor);

    rewriter.replaceOp(mm, resultTensor);
    return success();
  }
};

namespace {
class ConvertTorchToAnalog
    : public ConvertTorchToAnalogBase<ConvertTorchToAnalog> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<Torch::TorchDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<
        func::FuncDialect, 
        Torch::TorchDialect,
        linalg::LinalgDialect,
        arith::ArithDialect,
        bufferization::BufferizationDialect>();

    target.addIllegalOp<AtenMmOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });


    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    patterns.add<MmToCall>(typeConverter, context);

    // Add Patterns

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToAnalogPass() {
  return std::make_unique<ConvertTorchToAnalog>();
}
