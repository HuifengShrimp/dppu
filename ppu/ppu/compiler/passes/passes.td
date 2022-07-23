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

// pphlo passes

include "mlir/Pass/PassBase.td"

def HloLegalizeToPPHloPass : Pass<"hlo-legalize-to-pphlo", "ModuleOp"> {
  let summary = "Legalize from HLO dialect to pphlo dialect.";
  let constructor = "createLegalizeToPPHloPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
  let options = [
    Option<"io_visibility_json_", "io-visibility-json", "std::string", "",
           "io visibility of this module">,
  ];
}

def DecomposeComparison : FunctionPass<"decompose-comparison"> {
  let summary = "Decompose comparisons into basic ops.";
  let constructor = "createDecomposeComparisonPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def DecomposeSelect : FunctionPass<"decompose-select"> {
  let summary = "Decompose select into basic ops.";
  let constructor = "createDecomposeSelectPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def DecomposeDivide : FunctionPass<"decompose-divide"> {
  let summary = "Decompose divide into basic ops.";
  let constructor = "createDecomposeDividePass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def DecomposeSqrt : FunctionPass<"decompose-sqrt"> {
  let summary = "Decompose sqrt into basic ops.";
  let constructor = "createDecomposeSqrtPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}

def LowerConversionCast : FunctionPass<"lower-conversion-cast"> {
  let summary = "Lower UnrealizedConversionCastOp created during dialect conversion.";
  let constructor = "createLowerConversionCastPass()";
  let dependentDialects = ["pphlo::PPHloDialect"];
}
