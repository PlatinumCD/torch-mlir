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

#include "mlir/IR/Types.h"                 // for mlir::Type and its print()

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ConvToCall : public OpConversionPattern<AtenConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AtenConvolutionOp conv, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = conv.getLoc();

    // Get operands
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();  // Assuming bias is always present for simplicity
    Value stride = adaptor.getStride();
    Value padding = adaptor.getPadding();
    Value dilation = adaptor.getDilation();
    Value transposed = adaptor.getTransposed();
    Value output_padding = adaptor.getOutputPadding();
    Value groups = adaptor.getGroups();

    // Unpack lists using torch::PrimListUnpackOp (assuming 2D convolution)
    Type torchIntTy = torch::Torch::IntType::get(conv->getContext());
    SmallVector<Type, 2> intTypes = {torchIntTy, torchIntTy};

    auto strideUnpacked = rewriter.create<torch::Torch::PrimListUnpackOp>(loc, intTypes, stride);
    Value stride_h = strideUnpacked.getResult(0);
    Value stride_w = strideUnpacked.getResult(1);

    auto paddingUnpacked = rewriter.create<torch::Torch::PrimListUnpackOp>(loc, intTypes, padding);
    Value pad_h = paddingUnpacked.getResult(0);
    Value pad_w = paddingUnpacked.getResult(1);

    auto dilationUnpacked = rewriter.create<torch::Torch::PrimListUnpackOp>(loc, intTypes, dilation);
    Value dilation_h = dilationUnpacked.getResult(0);
    Value dilation_w = dilationUnpacked.getResult(1);


    Value output_pad_h, output_pad_w;
    if (auto listConstruct = dyn_cast_or_null<torch::Torch::PrimListConstructOp>(output_padding.getDefiningOp())) {
      if (listConstruct.getOperands().size() >= 2) {
        auto outputPadUnpacked = rewriter.create<torch::Torch::PrimListUnpackOp>(loc, intTypes, output_padding);
        output_pad_h = outputPadUnpacked.getResult(0);
        output_pad_w = outputPadUnpacked.getResult(1);
      } else {
        // Use default 0 for empty or invalid lists
        output_pad_h = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        output_pad_w = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      }
    } else {
      // Fallback for non-ListConstruct output_padding
      output_pad_h = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      output_pad_w = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    }

    // Convert !torch.int to i64
    Type i64Ty = rewriter.getI64Type();
    Value stride_h_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, stride_h);
    Value stride_w_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, stride_w);
    Value pad_h_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, pad_h);
    Value pad_w_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, pad_w);
    Value dilation_h_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, dilation_h);
    Value dilation_w_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, dilation_w);

    Value output_pad_h_i64 = output_pad_h; // Already i64 if from arith.constant
    Value output_pad_w_i64 = output_pad_w; // Already i64 if from arith.constant
    if (output_pad_h.getType().isa<torch::Torch::IntType>()) {
      output_pad_h_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, output_pad_h);
      output_pad_w_i64 = rewriter.create<torch::TorchConversion::ToI64Op>(loc, i64Ty, output_pad_w);
    }

    // Cast i64 to i32
    Type i32Ty = rewriter.getI32Type();
    Value stride_h_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, stride_h_i64);
    Value stride_w_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, stride_w_i64);
    Value pad_h_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, pad_h_i64);
    Value pad_w_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, pad_w_i64);
    Value dilation_h_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, dilation_h_i64);
    Value dilation_w_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, dilation_w_i64);
    Value output_pad_h_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, output_pad_h_i64);
    Value output_pad_w_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, output_pad_w_i64);
    Value groups_i32 = rewriter.create<arith::TruncIOp>(loc, i32Ty, groups);

    // Get output type
    Type newResTy = getTypeConverter()->convertType(conv.getType());
    auto rankedTy = newResTy.cast<RankedTensorType>();
    Type elemTy = rankedTy.getElementType();

    // --- build buffers ----------------------------------------------------------
    auto inputTy  = input.getType().cast<RankedTensorType>();
    auto weightTy = weight.getType().cast<RankedTensorType>();

    MemRefType inputMR  = MemRefType::get(inputTy.getShape(),  elemTy);
    MemRefType weightMR = MemRefType::get(weightTy.getShape(), elemTy);
    MemRefType outMR    = MemRefType::get(rankedTy.getShape(),  elemTy);

    Value inputBuf  = rewriter.create<bufferization::ToMemrefOp>(loc, inputMR,  input);
    Value weightBuf = rewriter.create<bufferization::ToMemrefOp>(loc, weightMR, weight);
    Value outBuf    = rewriter.create<memref::AllocOp>(loc, outMR);

    // --- helper: memref → !llvm.ptr<f32> ---------------------------------------
    Type llvmPtrF32Ty = LLVM::LLVMPointerType::get(elemTy);
    auto extractPointer = [&](Value buf) -> Value {
      Value baseIdx  = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buf);
      auto  meta     = rewriter.create<memref::ExtractStridedMetadataOp>(loc, buf);
      Value offElems = meta.getOffset();
      Value baseI64  = rewriter.create<arith::IndexCastOp>(loc, i64Ty, baseIdx);
      Value elemB    = rewriter.create<arith::ConstantIntOp>(loc, 4, 64);            // sizeof(f32)
      Value offB     = rewriter.create<arith::MulIOp>(loc,
                          rewriter.create<arith::IndexCastOp>(loc, i64Ty, offElems),
                          elemB);
      Value ptrI64   = rewriter.create<arith::AddIOp>(loc, baseI64, offB);
      return rewriter.create<LLVM::IntToPtrOp>(loc, llvmPtrF32Ty, ptrI64);
    };

    // --- raw pointers -----------------------------------------------------------
    Value inputPtr  = extractPointer(inputBuf);
    Value weightPtr = extractPointer(weightBuf);
    Value outPtr    = extractPointer(outBuf);

    // --- bias handling ----------------------------------------------------------
    Value biasPtr; // !llvm.ptr<f32>
    if (bias.getType().isa<torch::Torch::NoneType>()) {
      biasPtr = rewriter.create<LLVM::NullOp>(loc, llvmPtrF32Ty);       // nullptr
    } else {
      auto biasTy  = bias.getType().cast<RankedTensorType>();
      MemRefType bMR = MemRefType::get(biasTy.getShape(), elemTy);
      Value biasBuf  = rewriter.create<bufferization::ToMemrefOp>(loc, bMR, bias);
      biasPtr        = extractPointer(biasBuf);
    }


    // Init outputu buffer to zero
    Value zeroVal = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elemTy, 0.0));
    rewriter.create<linalg::FillOp>(loc, ValueRange{zeroVal}, ValueRange{outBuf});

    // Constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value c3 = rewriter.create<arith::ConstantIndexOp>(loc, 3);

    // Extract dimensions
    Value N = rewriter.create<memref::DimOp>(loc, inputBuf, c0);
    Value C_in = rewriter.create<memref::DimOp>(loc, inputBuf, c1);
    Value H = rewriter.create<memref::DimOp>(loc, inputBuf, c2);
    Value W = rewriter.create<memref::DimOp>(loc, inputBuf, c3);
    Value C_out = rewriter.create<memref::DimOp>(loc, weightBuf, c0);
    Value Kh = rewriter.create<memref::DimOp>(loc, weightBuf, c2);
    Value Kw = rewriter.create<memref::DimOp>(loc, weightBuf, c3);

    // Cast dimensions to i32
    Value N_i32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, N);
    Value C_in_i32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, C_in);
    Value H_i32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, H);
    Value W_i32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, W);
    Value C_out_i32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, C_out);
    Value Kh_i32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, Kh);
    Value Kw_i32 = rewriter.create<arith::IndexCastOp>(loc, i32Ty, Kw);

 
    // Ensure function declaration
    ModuleOp mod = conv->getParentOfType<ModuleOp>();
    if (!mod.lookupSymbol<func::FuncOp>("analog_conv")) {
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(mod.getBody());
      auto fnTy = rewriter.getFunctionType(
          {llvmPtrF32Ty, llvmPtrF32Ty, llvmPtrF32Ty, llvmPtrF32Ty, // input, weight, bias, output
           i32Ty, i32Ty, i32Ty, i32Ty, // N, C_in, H, W
           i32Ty, i32Ty, i32Ty, // C_out, Kh, Kw
           i32Ty, i32Ty, // stride_h, stride_w
           i32Ty, i32Ty, // pad_h, pad_w
           i32Ty, i32Ty, // dilation_h, dilation_w
           i32Ty, // groups
           rewriter.getI1Type(), // transposed
           i32Ty, i32Ty}, // output_pad_h, output_pad_w
          {});
      rewriter.create<func::FuncOp>(loc, "analog_conv", fnTy).setPrivate();
    }

    // Create function call
    rewriter.create<func::CallOp>(
        loc, "analog_conv", TypeRange{},
        ValueRange{inputPtr, weightPtr, biasPtr, outPtr,
                   N_i32, C_in_i32, H_i32, W_i32,
                   C_out_i32, Kh_i32, Kw_i32,
                   stride_h_i32, stride_w_i32,
                   pad_h_i32, pad_w_i32,
                   dilation_h_i32, dilation_w_i32,
                   groups_i32,
                   transposed,
                   output_pad_h_i32, output_pad_w_i32});

    // Convert output memref back to tensor
    Value outTensor = rewriter.create<bufferization::ToTensorOp>(loc, outBuf);
    outTensor.getDefiningOp()->setAttr("restrict", rewriter.getUnitAttr());
    outTensor = rewriter.create<tensor::CastOp>(loc, newResTy, outTensor);
    rewriter.replaceOp(conv, outTensor);

    return success();
  }
};


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
  registry.insert<LLVM::LLVMDialect>();
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
      LLVM::LLVMDialect,
      arith::ArithDialect,
      bufferization::BufferizationDialect>();
  
  target.addIllegalOp<AtenMmOp>();
  target.addIllegalOp<AtenConvolutionOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  
  
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  TorchConversion::setupBackendTypeConversion(target, typeConverter);
  
  RewritePatternSet patterns(context);
  patterns.add<MmToCall>(typeConverter, context);
  patterns.add<ConvToCall>(typeConverter, context);
  
  // Add Patterns
  
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
  }
};

} // namespace
  
std::unique_ptr<OperationPass<func::FuncOp>>  mlir::torch::createConvertTorchToAnalogPass() {
    return std::make_unique<ConvertTorchToAnalog>();
}
