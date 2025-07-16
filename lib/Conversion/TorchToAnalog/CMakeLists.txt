add_mlir_conversion_library(TorchMLIRTorchToAnalog
  TorchToAnalog.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToAnalog

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRBufferizationDialect       # ← add this
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRTorchToAnalog)
