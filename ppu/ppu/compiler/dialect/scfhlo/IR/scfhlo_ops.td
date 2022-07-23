//
// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef SCFHLO_OPS
#define SCFHLO_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SCFHLO_Dialect : Dialect {
  let name = "scfhlo";
  let cppNamespace = "::mlir::scfhlo";
}

class SCFHLO_Op<string mnemonic, list<OpTrait> traits>
    : Op<SCFHLO_Dialect, mnemonic, traits> {
}

def SCFHLO_IfOp : SCFHLO_Op<"if", [
  AttrSizedOperandSegments, RecursiveSideEffects,
  SingleBlockImplicitTerminator<"mhlo::ReturnOp">
]> {
  let summary = "If operator";

  let arguments = (ins AnyType:$condition,
                       Variadic<AnyType>:$trueDestOperands,
                       Variadic<AnyType>:$falseDestOperands);

  let regions = (region SizedRegion<1>:$true_branch,
                        SizedRegion<1>:$false_branch);

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 }
    ;
    // Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }
    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }
    unsigned getNumTrueOperands() { return getTrueOperands().size(); }
    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index) {
      trueDestOperandsMutable().erase(index);
    }
    // Accessors for operands to the 'false' destination.
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }
    operand_range getTrueOperands() { return trueDestOperands(); }
    operand_range getFalseOperands() { return falseDestOperands(); }
    unsigned getNumFalseOperands() { return getFalseOperands().size(); }
    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      falseDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }
    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];
}

#endif
