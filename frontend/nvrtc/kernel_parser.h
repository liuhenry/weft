#ifndef WEFT_FRONTEND_NVRTC_KERNEL_PARSER_H
#define WEFT_FRONTEND_NVRTC_KERNEL_PARSER_H

#include <clang/AST/Decl.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Type.h>

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace weft::nvrtc {

class Param {
 public:
  Param(const clang::ParmVarDecl* parm);

  constexpr size_t size() const noexcept { return size_; }
  constexpr bool is_pointer() const noexcept { return is_pointer_; }
  constexpr bool is_const() const noexcept { return is_const_; }

  friend std::ostream& operator<<(std::ostream& os, const Param& param);
  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const Param& param);

 private:
  std::string qualified_name_;
  std::string type_;
  size_t size_;
  bool is_pointer_;
  bool is_const_;
};

template <typename T>
using metadata_t = std::unordered_map<T, std::shared_ptr<std::vector<Param>>>;

class KernelVisitor : public clang::RecursiveASTVisitor<KernelVisitor> {
 public:
  KernelVisitor(metadata_t<std::string>& metadata) : metadata_{metadata} {}
  bool VisitFunctionDecl(clang::FunctionDecl* func);

 private:
  metadata_t<std::string>& metadata_;
};

class KernelMetadata {
 public:
  KernelMetadata() : kernel_visitor_{metadata_} {}

  const std::shared_ptr<std::vector<Param>>& at(std::string name) const {
    return metadata_.at(name);
  }

  const std::shared_ptr<std::vector<Param>>& at(uint64_t handle) const {
    return metadata_handle_.at(handle);
  }

  auto emplace(uint64_t function_handle,
               const std::shared_ptr<std::vector<Param>>& vec_ptr) {
    return metadata_handle_.emplace(function_handle, vec_ptr);
  }

  void parse_cu(std::string src);

 private:
  KernelVisitor kernel_visitor_;
  metadata_t<std::string> metadata_;
  metadata_t<uint64_t> metadata_handle_;
};

}  // namespace weft::nvrtc

#endif  // WEFT_FRONTEND_NVRTC_KERNEL_PARSER_H
