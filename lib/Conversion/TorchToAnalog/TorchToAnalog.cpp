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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"


#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

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
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AtenMmOp mm, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = mm.getLoc();

    //--- 1. Grab operands -------------------------------------
    Value lhs = adaptor.getSelf();   // tensor<MxKxf32>
    Value rhs = adaptor.getMat2();   // tensor<KxNxf32>

    //--- 2. Result tensor type --------------------------------
    Type newResTy = getTypeConverter()->convertType(mm.getType());
    auto rankedTy = newResTy.cast<RankedTensorType>();
    Type elemTy   = rankedTy.getElementType();

    //--- 3. Ranked memrefs (keep them ranked!) ---------------
    auto lhsRTy = lhs.getType().cast<RankedTensorType>();
    auto rhsRTy = rhs.getType().cast<RankedTensorType>();
    auto outRTy = rankedTy;
    auto lhsMR  = MemRefType::get(lhsRTy.getShape(), elemTy);
    auto rhsMR  = MemRefType::get(rhsRTy.getShape(), elemTy);
    auto outMR  = MemRefType::get(outRTy.getShape(), elemTy);

    Value lhsBuf = rewriter.create<bufferization::ToMemrefOp>(loc, lhsMR, lhs);
    Value rhsBuf = rewriter.create<bufferization::ToMemrefOp>(loc, rhsMR, rhs);
    Value outBuf = rewriter.create<memref::AllocOp>(loc, outMR);   // scratch dest

    //--- 4. Zero dest because custom_mm does "+=" -----------------------
    Value zeroVal = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elemTy, 0.0));
    // linalg.fill(memref form) - in older MLIR use tensor Empty+Fill+Copy; 
    rewriter.create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{outBuf});

    //--- 5. Dynamic dims + row-major leading dims ------------------------------
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value M = rewriter.create<memref::DimOp>(loc, lhsBuf, c0);
    Value K = rewriter.create<memref::DimOp>(loc, lhsBuf, c1);
    Value N = rewriter.create<memref::DimOp>(loc, rhsBuf, c1);
    Value lda = K, ldb = N, ldc = N;

    //--- 6. Ensure raw-pointer func decl exists --------------------------------
    ModuleOp mod = mm->getParentOfType<ModuleOp>();
    auto i32Ty        = rewriter.getI32Type();
    auto llvmPtrF32Ty = LLVM::LLVMPointerType::get(elemTy);

    auto fnTy = rewriter.getFunctionType(
      {llvmPtrF32Ty, llvmPtrF32Ty, llvmPtrF32Ty,
       i32Ty,i32Ty,i32Ty,i32Ty,i32Ty,i32Ty},
      {});

    if (auto old = mod.lookupSymbol<func::FuncOp>("analog_mm")) {
      if (old.getFunctionType() != fnTy) {
        old.erase();
      }
    }

    if (!mod.lookupSymbol<func::FuncOp>("analog_mm")) {
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      rewriter.create<func::FuncOp>(loc, "analog_mm", fnTy).setPrivate();
    }

    //--- 7. Ranked → raw pointer (aligned, include offset) ---------------------
    auto i64Ty = rewriter.getI64Type();

    // aligned base ptrs (as index)
    Value lhsBaseIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, lhsBuf);
    Value rhsBaseIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, rhsBuf);
    Value outBaseIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, outBuf);

    // element offset in *elements*:
    // extract_strided_metadata returns (base, offset, sizes..., strides...)
    auto metaL = rewriter.create<memref::ExtractStridedMetadataOp>(loc, lhsBuf);
    auto metaR = rewriter.create<memref::ExtractStridedMetadataOp>(loc, rhsBuf);
    auto metaO = rewriter.create<memref::ExtractStridedMetadataOp>(loc, outBuf);
    Value lhsOffElems = metaL.getOffset();
    Value rhsOffElems = metaR.getOffset();
    Value outOffElems = metaO.getOffset();

    // convert base index -> i64
    Value lhsBaseI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, lhsBaseIdx);
    Value rhsBaseI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, rhsBaseIdx);
    Value outBaseI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, outBaseIdx);

    // scale offsets by sizeof(elem) (4 for f32) if non-zero
    Value elemBytes = rewriter.create<arith::ConstantIntOp>(loc, /*value=*/4, /*width=*/64);
    auto mulBytes = [&](Value offElems) -> Value {
      return rewriter.create<arith::MulIOp>(loc, rewriter.create<arith::IndexCastOp>(loc, i64Ty, offElems), elemBytes);
    };

    Value lhsOffB = mulBytes(lhsOffElems);
    Value rhsOffB = mulBytes(rhsOffElems);
    Value outOffB = mulBytes(outOffElems);

    // add base+offset
    Value lhsPtrI64 = rewriter.create<arith::AddIOp>(loc, lhsBaseI64, lhsOffB);
    Value rhsPtrI64 = rewriter.create<arith::AddIOp>(loc, rhsBaseI64, rhsOffB);
    Value outPtrI64 = rewriter.create<arith::AddIOp>(loc, outBaseI64, outOffB);

    // inttoptr to !llvm.ptr<f32>
    Value lhsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrF32Ty, lhsPtrI64);
    Value rhsPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrF32Ty, rhsPtrI64);
    Value outPtr = rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrF32Ty, outPtrI64);

    //--- 8. Cast dims/ld* to i32 ------------------------------------------------
    auto cast32 = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(loc, i32Ty, v);
    };

    Value Mi   = cast32(M);
    Value Ki   = cast32(K);
    Value Ni   = cast32(N);
    Value ldai = cast32(lda);
    Value ldbi = cast32(ldb);
    Value ldci = cast32(ldc);

    //--- 9. Call raw-pointer analog_mm -----------------------------------------
    rewriter.create<func::CallOp>(
      loc, "analog_mm", TypeRange{},
      ValueRange{lhsPtr, rhsPtr, outPtr,
                 Mi, Ki, Ni, ldai, ldbi, ldci});

    //--- 10. Return tensor aliasing outBuf -------------------------------------
    Value resultTensor = rewriter.create<bufferization::ToTensorOp>(loc, outBuf);
    resultTensor.getDefiningOp()->setAttr("restrict", rewriter.getUnitAttr());
    resultTensor = rewriter.create<tensor::CastOp>(loc, newResTy, resultTensor);
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
