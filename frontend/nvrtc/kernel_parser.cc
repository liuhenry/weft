#include "nvrtc/kernel_parser.h"

#include <clang/AST/Decl.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Tooling/Tooling.h>

#include <iostream>
#include <ostream>

namespace weft::nvrtc {

Param::Param(const clang::ParmVarDecl* parm)
    : qualified_name_{parm->getQualifiedNameAsString()},
      type_{parm->getOriginalType().getAsString()},
      size_{parm->getASTContext()
                .getTypeSizeInChars(parm->getOriginalType())
                .getQuantity()},
      is_pointer_{parm->getOriginalType()->isPointerType()},
      is_const_{is_pointer_ &&
                parm->getOriginalType()->getPointeeType().isConstQualified()} {}

bool KernelVisitor::VisitFunctionDecl(clang::FunctionDecl* func) {
  auto name = func->getNameInfo().getName().getAsString();
  if (func->getNumParams()) {
    metadata_.emplace(name, std::make_shared<std::vector<Param>>());
    for (auto const& parm : func->parameters()) {
      // Exclude _weft parameters
      if (parm->getQualifiedNameAsString().rfind("_weft", 0)) {
        metadata_.at(name)->emplace_back(parm);
      }
    }
  }
  return true;
}

void KernelMetadata::parse_cu(std::string src) {
  // Generate AST from source with clang
  // TODO: is there .cu support (CUDA C++ extension parsing errors)
  std::unique_ptr<clang::ASTUnit> ast(clang::tooling::buildASTFromCode(src));
  auto* decl = ast->getASTContext().getTranslationUnitDecl();
  if (decl) {
    llvm::errs() << "---------clang dump begin----------\n";
    decl->dump();
    llvm::errs() << "---------clang dump end----------\n";

    llvm::errs() << "---------AST KernelVisitor traversal begin----------\n";
    kernel_visitor_.TraverseDecl(decl);
    for (const auto& item : metadata_) {
      llvm::errs() << item.first << "\n";
      for (const auto& param : *item.second) {
        llvm::errs() << "\t" << param << "\n";
      }
    }
    llvm::errs() << "---------AST KernelVisitor traversal end----------\n";
  }
}

std::ostream& operator<<(std::ostream& os, const weft::nvrtc::Param& param) {
  os << param.qualified_name_ << " - " << param.type_
     << ", size: " << param.size_ << ", pointer: " << param.is_pointer_
     << ", const: " << param.is_const_;
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                              const weft::nvrtc::Param& param) {
  os << param.qualified_name_ << " - " << param.type_
     << ", size: " << param.size_ << ", pointer: " << param.is_pointer_
     << ", const: " << param.is_const_;
  return os;
}

}  // namespace weft::nvrtc
