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


#include "ppu/device/pphlo_executor.h"

#include <algorithm>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "ppu/core/shape_util.h"
#include "ppu/device/frame.h"
#include "ppu/dialect/pphlo_types.h"
#include "ppu/hal/hal.h"
#include "ppu/hal/polymorphic.h"
#include "ppu/hal/test_util.h"
#include "ppu/hal/value.h"
#include "ppu/utils/exception.h"

namespace ppu::device {

namespace {

std::vector<int64_t> build_shape(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> ret(shape.size());

  for (size_t idx = 0; idx < ret.size(); ++idx) {
    ret[idx] = shape[idx];
  }

  return ret;
}

template <typename T>
std::vector<T> build_vec_idx(const mlir::DenseIntElementsAttr &attr) {
  std::vector<T> ret;

  for (const auto &v : attr) {
    ret.emplace_back(static_cast<T>(v.getLimitedValue()));
  }

  return ret;
}

PtType getPtType(const mlir::Type &type) {
  if (auto ft = type.dyn_cast<mlir::FloatType>()) {
    switch (ft.getWidth()) {
    case 32:
      return PT_F32;
    case 64:
      return PT_F64;
    }
  }
  if (auto it = type.dyn_cast<mlir::IntegerType>()) {
    if (it.getWidth() == 1) {
      return PT_BOOL;
    }
    // In mlir, isSigned is for si[1-9][0-9]* type, isUnsigned is for
    // ui[1-9][0-9]*, i[1-9][0-9]* is signless IntegerType... So here, we only
    // check for isUnsigned, signless we treat it as signed.
    // See https://reviews.llvm.org/D72533
    switch (it.getWidth()) {
    case 8:
      return it.isUnsigned() ? PT_U8 : PT_I8;
    case 16:
      return it.isUnsigned() ? PT_U16 : PT_I16;
    case 32:
      return it.isUnsigned() ? PT_U32 : PT_I32;
    case 64:
      return it.isUnsigned() ? PT_U64 : PT_I64;
    }
  }
  PPU_THROW("Hit unknown pt_type");
}

// All sorts of gather helper functions
hal::Value reshapedGatherIndices(HalContext *ctx, int64_t index_vector_dim,
                                 const hal::Value &start_indices) {

  if (start_indices.shape().size() != static_cast<size_t>(index_vector_dim)) {
    return start_indices;
  }

  auto new_shape = start_indices.shape();
  new_shape.push_back(1);

  return hal::reshape(ctx, start_indices, new_shape);
}

struct IndexIterationSpace {
  std::vector<int64_t> index_base;
  std::vector<int64_t> index_count;
  std::vector<int64_t> index_incr;
};

// Returns an IndexIterationSpace that iterates over the output batch
// dimensions while keeping the rest of the output dimensions clamped to 0.
IndexIterationSpace iterationSpaceForOutputBatchIndices(
    llvm::ArrayRef<int64_t> output_shape,
    const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers) {
  int64_t output_rank = output_shape.size();
  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count;
  index_count.reserve(output_rank);

  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_batch_dim =
        !std::binary_search(dim_numbers.getOffsetDims().begin(),
                            dim_numbers.getOffsetDims().end(), i);
    index_count.push_back(is_output_batch_dim ? output_shape[i] : 1);
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(output_rank, 1)};
}

// Return an IndexIterationSpace that iterates over the output slice
// dimensions while keeping the rest of the output dimensions clamped to 0.
IndexIterationSpace iterationSpaceForOutputOffsetIndices(
    int64_t output_rank, const mlir::DenseIntElementsAttr &slice_sizes,
    const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers) {

  std::vector<int64_t> index_base(output_rank, 0);
  std::vector<int64_t> index_count(output_rank, 1);
  int64_t slice_sizes_idx = 0;

  for (int64_t i = 0; i < output_rank; i++) {
    bool is_output_window_dim =
        std::binary_search(dim_numbers.getOffsetDims().begin(),
                           dim_numbers.getOffsetDims().end(), i);
    if (is_output_window_dim) {
      while (std::binary_search(dim_numbers.getCollapsedSliceDims().begin(),
                                dim_numbers.getCollapsedSliceDims().end(),
                                slice_sizes_idx)) {
        slice_sizes_idx++;
      }
      index_count[i] =
          *(slice_sizes.getValues<int64_t>().begin() + slice_sizes_idx++);
    }
  }

  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(output_rank, 1)};
}

// This functor computes the contribution of start_indices to an input index
// corresponding to an output index.  That is, given an output index I, it
// picks out the batch indices in I and uses them to look up a starting index,
// G, from the start indices tensor, and expands G into the input space
// according to start_index_map.
class OutputBatchIndexToInputIndex {
public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputBatchIndexToInputIndex(
      const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers,
      llvm::ArrayRef<int64_t> input_shape, llvm::ArrayRef<int64_t> output_shape,
      const xt::xarray<int64_t> &start_indices)
      : dim_numbers_(dim_numbers), start_indices_(start_indices) {

    for (int64_t i = 0; i < static_cast<int64_t>(output_shape.size()); ++i) {
      output_dim_is_batch_dims_.push_back(
          !std::binary_search(dim_numbers_.getOffsetDims().begin(),
                              dim_numbers_.getOffsetDims().end(), i));
    }

    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); ++i) {
      int64_t index_of_input_dim_in_index_vector =
          std::distance(dim_numbers_.getStartIndexMap().begin(),
                        std::find(dim_numbers_.getStartIndexMap().begin(),
                                  dim_numbers_.getStartIndexMap().end(), i));

      if (static_cast<size_t>(index_of_input_dim_in_index_vector) ==
          dim_numbers_.getStartIndexMap().size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(start_indices_.shape().size());
    input_index_.resize(input_shape.size());
    int64_t index_vector_size =
        start_indices_.shape()[dim_numbers_.getIndexVectorDim()];
    index_vector_.resize(index_vector_size);

    start_indices_shape.reserve(start_indices_.shape().size());
    for (const auto &d : start_indices_.shape()) {
      start_indices_shape.emplace_back(static_cast<int64_t>(d));
    }
  }

  // Returns the contribution of start_indices to the input index
  // corresponding to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually  a stateless transformation from output_index to the
  // gather input index, but:
  //
  //  - Instead of allocating memory to represent the gather input index on
  //    every invocation we reuse the same storage for the result
  //    (input_index_), mutating it in place.
  //  - Instead of allocating buffers for temporary values like
  //    index_vector_index_ and index_vector on every invocation, we reuse the
  //    same storage for all invocations.
  //
  // This returns a Span into memory owned by the class.
  llvm::ArrayRef<int64_t> operator()(llvm::ArrayRef<int64_t> output_index) {
    propagateOutputIndexGatherDimsToIndexVectorIndex(output_index);
    fetchIndexVector();
    propagateIndexVectorToInputIndex();
    return input_index_;
  }

private:
  // Propagates the batch dimensions from the output index into
  // index_vector_index_ by mutating index_vector_index_ in place.  Does not
  // update the dim_numbers.index_vector_dim() dimension -- that's the
  // dimension we iterate over in FetchIndexVector.
  void propagateOutputIndexGatherDimsToIndexVectorIndex(
      llvm::ArrayRef<int64_t> output_index) {
    int64_t index_vector_index_i = 0;
    for (int64_t i = 0, e = output_index.size(); i < e; i++) {
      if (!output_dim_is_batch_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == dim_numbers_.getIndexVectorDim()) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = output_index[i];
    }
  }

  // Populates index_vector_ by iterating over start_indices_ according to
  // index_vector_index_.
  void fetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.getIndexVectorDim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      index_vector_[i] =
          start_indices_
              .data()[flattenIndex(index_vector_index_, start_indices_shape)];
    }
  }

  // Populates input_index_.
  void propagateIndexVectorToInputIndex() {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_index_vector_[i] != -1) {
        input_index_[i] = index_vector_[input_dim_value_to_index_vector_[i]];
      }
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the index vector.  See
  // PropagateIndexVectorToInputIndex.
  std::vector<int64_t> input_dim_value_to_index_vector_;

  // output_dim_is_batch_dims_[i] is true iff the output index i is a gather
  // dimension.
  std::vector<bool> output_dim_is_batch_dims_;

  // The buffer into which we construct an index into start_indices_ to fetch
  // the index vector.
  std::vector<int64_t> index_vector_index_;

  // The index vector fetched from start_indices_.
  std::vector<int64_t> index_vector_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;

  const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers_;
  const xt::xarray<int64_t> &start_indices_;
  std::vector<int64_t> start_indices_shape;
};

// This functor computes the contribution of the offset indices in an output
// index to an input index.  That is, given an output index I it picks out the
// output offset indices in I and expands it into an index into the input
// shape.
class OutputOffsetIndexToInputIndex {
public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit OutputOffsetIndexToInputIndex(
      const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers,
      llvm::ArrayRef<int64_t> input_shape,
      llvm::ArrayRef<int64_t> output_shape) {

    std::vector<int64_t> window_index_to_output_index;
    int64_t output_index_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(output_shape.size()); i++) {
      if (std::binary_search(dim_numbers.getOffsetDims().begin(),
                             dim_numbers.getOffsetDims().end(), i)) {
        window_index_to_output_index.push_back(output_index_count++);
      } else {
        output_index_count++;
      }
    }

    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.size()); i++) {
      if (std::binary_search(dim_numbers.getCollapsedSliceDims().begin(),
                             dim_numbers.getCollapsedSliceDims().end(), i)) {
        input_dim_value_to_output_index_.push_back(-1);
      } else {
        input_dim_value_to_output_index_.push_back(
            window_index_to_output_index[window_dim_count++]);
      }
    }

    input_index_.resize(input_shape.size());
  }

  // Returns the contribution of the window indices to the input index
  // corresponding to output_index.  See gather_inner_loop_body.
  //
  // This is conceptually a stateless transformation from output_index to the
  // window input index, but instead of allocating memory to represent the
  // gather input index on every invocation we reuse the same storage for the
  // result (input_index_), mutating it in place.
  //
  // This returns a Span into memory owned by the class.
  llvm::ArrayRef<int64_t> operator()(llvm::ArrayRef<int64_t> output_index) {
    propagateOutputIndexWindowDimsToInputIndex(output_index);
    return input_index_;
  }

  // Returns for a given 'input_dim' the corresponding output dimension index,
  // or -1 if 'input_dim' is an elided window dimension.
  int64_t input_dim_value_to_output_index(int64_t input_dim) {
    return input_dim_value_to_output_index_[input_dim];
  }

private:
  // Propagates window dimensions from the output index to input_index_ by
  // mutating input_index_ in place.
  void propagateOutputIndexWindowDimsToInputIndex(
      llvm::ArrayRef<int64_t> output_index) {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_output_index_[i] != -1) {
        input_index_[i] = output_index[input_dim_value_to_output_index_[i]];
      }
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the output index. See
  // PropagateOutputIndexWindowDimsToInputIndex.
  std::vector<int64_t> input_dim_value_to_output_index_;

  // The result computed by this functor.  operator() returns a Span into
  // this vector.
  std::vector<int64_t> input_index_;
};

template <typename FnTy>
void forEachIndex(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> base,
                  llvm::ArrayRef<int64_t> count, llvm::ArrayRef<int64_t> incr,
                  FnTy &&visitor_function) {
  PPU_ENFORCE_EQ(shape.size(), base.size());
  PPU_ENFORCE_EQ(incr.size(), base.size());
  PPU_ENFORCE_EQ(count.size(), base.size());

  const auto rank = static_cast<int64_t>(shape.size());
  // Allows handling R0 arrays, such that the visitor function will be called
  // once with the proper empty indexes.
  int64_t n = -1;
  std::vector<int64_t> indexes(base.begin(), base.end());

  while (n < rank) {
    visitor_function(indexes);
    // Increments dimensions in minor to major order.
    for (n = 0; n < rank; ++n) {
      indexes[n] += incr[n];
      if (indexes[n] < base[n] + count[n]) {
        break;
      }
      indexes[n] = base[n];
    }
  }
}

template <typename FnType>
void forEachIndex(llvm::ArrayRef<int64_t> shape,
                  const FnType &visitor_function) {
  std::vector<int64_t> base(shape.size());
  std::vector<int64_t> incr(shape.size(), 1);
  return forEachIndex(shape, base,
                      /*count=*/shape, incr, visitor_function);
}

void performReductionStep(PPHloExecutor *executor,
                          llvm::ArrayRef<int64_t> input_index,
                          llvm::ArrayRef<int64_t> output_index,
                          llvm::ArrayRef<hal::Value> input_args,
                          llvm::ArrayRef<hal::Value> results,
                          mlir::Region &computation) {
  int num_args = results.size();

  std::vector<hal::Value> arg_values;
  arg_values.reserve(num_args);
  std::vector<hal::Value> accumulators;
  accumulators.reserve(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    arg_values.emplace_back(hal::make_value(
        executor->getContext(), input_args[i].vtype(),
        input_args[i].is_int() ? PtBufferView(0) : PtBufferView(0.0F)));
    accumulators.emplace_back(hal::make_value(
        executor->getContext(), input_args[i].vtype(),
        input_args[i].is_int() ? PtBufferView(0) : PtBufferView(0.0F)));

    arg_values[i].CopyElementFrom(input_args[i], input_index, {});
    accumulators[i].CopyElementFrom(results[i], output_index, {});
  }

  // Evaluate computation with specified literal operands.
  std::vector<hal::Value> embedded_operands;
  embedded_operands.reserve(computation.getNumArguments());
  for (const auto &accumulator : accumulators) {
    embedded_operands.push_back(accumulator);
  }
  for (const auto &local_input : arg_values) {
    embedded_operands.push_back(local_input);
  }

  auto computed_results =
      executor->executeRegion(computation, embedded_operands);

  for (int64_t i = 0; i < num_args; ++i) {
    results[i].CopyElementFrom(computed_results[i], {}, output_index);
  }
}

void generateReduceOutputElement(PPHloExecutor *executor,
                                 llvm::ArrayRef<int64_t> output_index,
                                 llvm::ArrayRef<hal::Value> init_values,
                                 llvm::ArrayRef<hal::Value> input_args,
                                 llvm::ArrayRef<hal::Value> results,
                                 mlir::Region &function,
                                 llvm::ArrayRef<int64_t> arg_dim_steps,
                                 llvm::ArrayRef<int64_t> arg_dim_counts,
                                 llvm::ArrayRef<int64_t> result_to_arg_index) {
  std::vector<int64_t> arg_dimensions = input_args[0].shape();
  std::vector<int64_t> base(arg_dimensions.size());
  for (int64_t i = 0; i < static_cast<int64_t>(output_index.size()); ++i) {
    base[result_to_arg_index[i]] = output_index[i];
  }

  for (int64_t i = 0; i < static_cast<int64_t>(results.size()); ++i) {
    results[i].CopyElementFrom(init_values[i], {}, output_index);
  }

  // Iterates only over reduced shape, as counts and steps are set to zero
  // for all non-reduced dimensions.
  forEachIndex(arg_dimensions, base, arg_dim_counts, arg_dim_steps,
               [&](llvm::ArrayRef<int64_t> input_index) {
                 return performReductionStep(executor, input_index,
                                             output_index, input_args, results,
                                             function);
               });
}

} // namespace

const hal::Value &PPHloExecutor::lookupValue(::mlir::Value v) const {
  for (auto iter = frames_.rbegin(); iter != frames_.rend(); ++iter) {
    const auto *frame = *iter;
    if (frame->hasValue(v)) {
      return frame->getValue(v);
    }
  }
  // Somehow cannot find this value on stack, print a reasonable error message.
  std::string str;
  llvm::raw_string_ostream debug_s(str);
  v.getDefiningOp()->print(debug_s);
  PPU_ENFORCE(false, "Try to get a non-exist value, defined at {}",
              debug_s.str());
}

void PPHloExecutor::executeVReduce(mlir::pphlo::ReduceOp &op) {
  int64_t num_args = op->getNumOperands() / 2;
  std::vector<int64_t> dimensions_to_reduce =
      build_vec_idx<int64_t>(op.dimensions());
  auto &function = op.body();

  llvm::SmallVector<hal::Value, 2> input_args(num_args);
  llvm::SmallVector<hal::Value, 2> init_values(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    input_args[i] = lookupValue(op.inputs()[i]);
    init_values[i] = lookupValue(op.init_values()[i]);
  }

  // All args and results have the same dimensions, so pick an arbitrary one.
  const auto &output_shape =
      op->getResultTypes()[0].dyn_cast<mlir::RankedTensorType>().getShape();

  std::vector<int64_t> arg_dimensions = input_args[0].shape();

  // All increments are set to 0.
  std::vector<int64_t> arg_dim_steps(arg_dimensions.size());

  // All counts are set to 0.
  std::vector<int64_t> arg_dim_counts(arg_dimensions.size());

  // Set steps and counts for reduced dimensions.
  // This avoids iterating over non-reduced dimensions, as their step
  // and count is set to zero.
  for (const int64_t dim : dimensions_to_reduce) {
    arg_dim_steps[dim] = 1;
    arg_dim_counts[dim] = arg_dimensions[dim];
  }

  // Map each dimension in the result to a dimension in arg that isn't
  // being reduced.
  std::vector<int64_t> result_to_arg_index;
  for (int64_t i = 0; i < static_cast<int64_t>(arg_dimensions.size()); ++i) {
    if (arg_dim_steps[i] == 0) {
      result_to_arg_index.push_back(i);
    }
  }

  llvm::SmallVector<hal::Value, 2> results(num_args);
  for (int64_t i = 0; i < num_args; ++i) {
    auto ret_shape =
        op->getResultTypes()[i].dyn_cast<mlir::RankedTensorType>().getShape();
    results[i] = hal::broadcast_to(ctx_,
                                   hal::make_value(ctx_, input_args[i].vtype(),
                                                   input_args[i].is_int()
                                                       ? PtBufferView(0)
                                                       : PtBufferView(0.0F)),
                                   build_shape(ret_shape));
  }

  forEachIndex(output_shape, [&](llvm::ArrayRef<int64_t> output_index) {
    return generateReduceOutputElement(
        this, output_index, init_values, input_args, results, function,
        arg_dim_steps, arg_dim_counts, result_to_arg_index);
  });

  for (int64_t i = 0; i < num_args; ++i) {
    getCurrentFrame()->addValue(op->getResult(i), results[i]);
  }
}

void PPHloExecutor::execute(mlir::pphlo::ReduceOp &op) {
  if (op->getNumOperands() > 2) {
    executeVReduce(op);
  } else {
    // Vectorized reduce can have simd lambda op in the middle, so disable type
    // checker
    auto old = config_.enable_type_checker;
    config_.enable_type_checker = false;
    getCurrentFrame()->addValue(
        op.getResult(0),
        hal::reduce(ctx_, lookupValue(op.inputs()[0]),
                    lookupValue(op.init_values()[0]),
                    build_vec_idx<size_t>(op.dimensions()),
                    [&](const hal::Value &a, const hal::Value &b) {
                      const auto &ret = executeRegion(op.body(), {a, b});
                      PPU_ENFORCE(ret.size() == 1);
                      return ret.front();
                    }));
    config_.enable_type_checker = old;
  }
}

bool PPHloExecutor::getConditionValue(const hal::Value &value) const {
  PPU_ENFORCE(value.numel() == 1, "Condition value must be a scalar tensor.");
  PPU_ENFORCE(value.is_int(), "Condition value must be an integer type.");
  PPU_ENFORCE(GetDecodeType(value.dtype()) == PT_I64,
              "Decoded type of INT should be int64.");

  hal::Value v;
  if (ctx_->rt_config().reveal_secret_condition()) {
    v = hal::reveal(ctx_, value);
  } else {
    PPU_ENFORCE(
        value.is_public(),
        "If op condition variable is not a public, either replace IfOp with a "
        "SelectOp or change config to allow reveal secret control flow.");
    v = value;
  }
  const auto public_val = hal::dump_public(ctx_, v);
  return (public_val.at<int64_t>({}) != 0);
}

void PPHloExecutor::execute(mlir::pphlo::IfOp &op) {
  bool v = getConditionValue(lookupValue(op.getOperand(0)));

  // Prepare a new frame anyway
  std::vector<hal::Value> inputs;
  if (v) {
    // True branch
    // Copy true args
    for (size_t cnt = 0; cnt < op.true_branch().getNumArguments(); ++cnt) {
      inputs.emplace_back(
          lookupValue(op.getOperand(cnt + 1))); // Offset the pred operand
    }
  } else {
    // False branch
    // Copy false args
    size_t trueBranchArgCnt = op.true_branch().getNumArguments();
    for (size_t cnt = 0; cnt < op.false_branch().getNumArguments(); ++cnt) {
      inputs.emplace_back(lookupValue(op.getOperand(
          cnt + 1 +
          trueBranchArgCnt))); // Offset the pred operand and true branch
    }
  }

  auto results =
      executeRegion(v ? op.true_branch() : op.false_branch(), inputs);

  // Copy output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    getCurrentFrame()->addValue(ret.value(), results[ret.index()]);
  }
}

/// While evalation order:
/// 1. Forward all args into cond block
/// 2. Evaluate condition
/// 3. If true -> run body with all args forward into body block
/// 4. If false -> done, set output
void PPHloExecutor::execute(mlir::pphlo::WhileOp &op) {
  // First inputs vectors
  std::vector<hal::Value> inputs;
  inputs.reserve(op->getNumOperands());

  // Prepare inputs
  for (const auto operand : op->getOperands()) {
    inputs.emplace_back(lookupValue(operand));
  }

  // Push frame
  auto eval_cond = [&](llvm::ArrayRef<hal::Value> inputs) -> bool {
    // Sanity inputs
    PPU_ENFORCE(inputs.size() == op.cond().getNumArguments());

    // Now evaluate cond
    auto ret = executeRegion(op.cond(), inputs);

    // Get cond result
    PPU_ENFORCE(ret.size() == 1,
                "WhileOp condition body should not return more than 1 result.");

    return getConditionValue(ret[0]);
  };

  while (eval_cond(inputs)) {
    // Sanity inputs
    PPU_ENFORCE(inputs.size() == op.body().getNumArguments());

    // dispatch body
    inputs = executeRegion(op.body(), inputs);
  }

  // Assign output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    getCurrentFrame()->addValue(ret.value(), inputs[ret.index()]);
  }
}

#define STANDARD_UNARY_OP_EXEC_IMPL(OpName, Fn)                                \
  void PPHloExecutor::execute(mlir::pphlo::OpName &op) {                       \
    getCurrentFrame()->addValue(op.getResult(),                                \
                                Fn(ctx_, lookupValue(op.getOperand())));       \
  }

STANDARD_UNARY_OP_EXEC_IMPL(ReciprocalOp, hal::reciprocal)
STANDARD_UNARY_OP_EXEC_IMPL(NegOp, hal::negate)
STANDARD_UNARY_OP_EXEC_IMPL(ExpOp, hal::exp)
STANDARD_UNARY_OP_EXEC_IMPL(LogOp, hal::log)
STANDARD_UNARY_OP_EXEC_IMPL(Log1pOp, hal::log1p)
STANDARD_UNARY_OP_EXEC_IMPL(FloorOp, hal::floor)
STANDARD_UNARY_OP_EXEC_IMPL(CeilOp, hal::ceil)
STANDARD_UNARY_OP_EXEC_IMPL(AbsOp, hal::abs)
STANDARD_UNARY_OP_EXEC_IMPL(LogisticOp, hal::logistic)
STANDARD_UNARY_OP_EXEC_IMPL(NotOp, hal::logical_not)
STANDARD_UNARY_OP_EXEC_IMPL(ProtectOp, hal::p2s)

#undef STANDARD_UNARY_OP_EXEC_IMPL

#define STANDARD_BINARY_OP_EXEC_IMPL(OpName, Fn)                               \
  void PPHloExecutor::execute(mlir::pphlo::OpName &op) {                       \
    getCurrentFrame()->addValue(                                               \
        op.getResult(),                                                        \
        Fn(ctx_, lookupValue(op.lhs()), lookupValue(op.rhs())));               \
  }

STANDARD_BINARY_OP_EXEC_IMPL(AddOp, hal::add)
STANDARD_BINARY_OP_EXEC_IMPL(EqualOp, hal::equal);
STANDARD_BINARY_OP_EXEC_IMPL(SubOp, hal::sub)
STANDARD_BINARY_OP_EXEC_IMPL(LessOp, hal::less)
STANDARD_BINARY_OP_EXEC_IMPL(GreaterOp, hal::greater)
STANDARD_BINARY_OP_EXEC_IMPL(MulOp, hal::mul)
STANDARD_BINARY_OP_EXEC_IMPL(PowOp, hal::power)
STANDARD_BINARY_OP_EXEC_IMPL(MaxOp, hal::max)
STANDARD_BINARY_OP_EXEC_IMPL(MinOp, hal::min)
STANDARD_BINARY_OP_EXEC_IMPL(AndOp, hal::bitwise_and)
STANDARD_BINARY_OP_EXEC_IMPL(OrOp, hal::bitwise_or)
STANDARD_BINARY_OP_EXEC_IMPL(XorOp, hal::bitwise_xor)

#undef STANDARD_BINARY_OP_EXEC_IMPL

#define LOWERED_OP_IMPL(OpName)                                                \
  void PPHloExecutor::execute(mlir::pphlo::OpName &) {                         \
    PPU_THROW("Lowered op should not occur at backend");                       \
  }

LOWERED_OP_IMPL(SqrtOp)
LOWERED_OP_IMPL(SelectOp)
LOWERED_OP_IMPL(RevealOp)
LOWERED_OP_IMPL(ReturnOp)
LOWERED_OP_IMPL(NotEqualOp)
LOWERED_OP_IMPL(LessEqualOp)
LOWERED_OP_IMPL(GreaterEqualOp)
LOWERED_OP_IMPL(DivOp)

#undef LOWERED_OP_IMPL

void PPHloExecutor::execute(mlir::pphlo::TransposeOp &op) {
  getCurrentFrame()->addValue(
      op.getResult(), hal::transpose(ctx_, lookupValue(op.getOperand()),
                                     build_vec_idx<size_t>(op.permutation())));
}

void PPHloExecutor::execute(mlir::pphlo::DotOp &op) {
  const auto &lhs = getCurrentFrame()->getValue(op.lhs()).shape();
  const auto &rhs = getCurrentFrame()->getValue(op.rhs()).shape();
  PPU_ENFORCE(!lhs.empty() && lhs.size() <= 2);
  PPU_ENFORCE(!rhs.empty() && rhs.size() <= 2);

  getCurrentFrame()->addValue(
      op.getResult(), hal::matmul(ctx_, getCurrentFrame()->getValue(op.lhs()),
                                  getCurrentFrame()->getValue(op.rhs())));
}

void PPHloExecutor::execute(mlir::pphlo::BroadcastOp &op) {
  auto to_shape =
      build_shape(op.getType().dyn_cast<mlir::RankedTensorType>().getShape());
  getCurrentFrame()->addValue(
      op.getResult(),
      hal::broadcast_to(ctx_, lookupValue(op.getOperand()), to_shape,
                        build_vec_idx<size_t>(op.broadcast_dimensions())));
}

void PPHloExecutor::execute(mlir::pphlo::ReshapeOp &op) {
  auto to_shape =
      build_shape(op.getType().dyn_cast<mlir::RankedTensorType>().getShape());
  getCurrentFrame()->addValue(
      op.getResult(),
      hal::reshape(ctx_, lookupValue(op.getOperand()), to_shape));
}

void PPHloExecutor::execute(mlir::pphlo::RngUniformOp &op) {
  auto to_shape =
      build_shape(op.getType().dyn_cast<mlir::RankedTensorType>().getShape());
  getCurrentFrame()->addValue(op.getResult(),
                              hal::rng_uniform(ctx_, lookupValue(op.a()),
                                               lookupValue(op.b()), to_shape));
}

void PPHloExecutor::execute(mlir::pphlo::ConvertOp &op) {
  auto dst_dtype =
      mlir::pphlo::TypeTools().isIntegerType(op.getType()) ? DT_INT : DT_FXP;
  getCurrentFrame()->addValue(
      op.getResult(),
      hal::cast_dtype(ctx_, lookupValue(op.getOperand()), dst_dtype));
}

void PPHloExecutor::execute(mlir::pphlo::ConstOp &op) {
  const auto &val = op.value();
  const auto &dea = val.dyn_cast<mlir::DenseElementsAttr>();
  const auto &type = val.getType().dyn_cast<mlir::RankedTensorType>();
  const auto &dst_shape = build_shape(type.getShape());
  const auto buf = makeBuffer(dea.getRawData().data(), dea.getRawData().size());

  if (dea.isSplat()) {
    auto scalar =
        hal::constant(ctx_, buf, getPtType(type.getElementType()), {});
    getCurrentFrame()->addValue(op.getResult(),
                                hal::broadcast_to(ctx_, scalar, dst_shape));
  } else {
    getCurrentFrame()->addValue(
        op.getResult(),
        hal::constant(ctx_, buf, getPtType(type.getElementType()), dst_shape));
  }
}

void PPHloExecutor::execute(mlir::pphlo::IotaOp &op) {
  const auto &ret_type =
      op.output().getType().dyn_cast<mlir::RankedTensorType>();
  const size_t numel = ret_type.getShape()[op.iota_dimension()];

  auto ret = hal::iota(ctx_, numel);

  if (type_tools_.isFxpType(ret_type)) {
    ret = hal::int2fxp(ctx_, ret);
  }

  if (ret_type.getShape().size() > 1) {
    // Need a broadcast
    ret = hal::broadcast_to(ctx_, ret, build_shape(ret_type.getShape()));
  }

  getCurrentFrame()->addValue(op.output(), std::move(ret));
}

void PPHloExecutor::execute(mlir::pphlo::ConcatenateOp &op) {
  std::vector<hal::Value> values;

  // Lookup values
  for (auto operand : op->getOperands()) {
    values.emplace_back(lookupValue(operand));
  }

  // set result
  getCurrentFrame()->addValue(op.getResult(),
                              hal::concatenate(ctx_, values, op.dimension()));
}

void PPHloExecutor::execute(mlir::pphlo::SliceOp &op) {
  getCurrentFrame()->addValue(
      op.getResult(), hal::slice(ctx_, lookupValue(op.getOperand()),
                                 build_vec_idx<size_t>(op.start_indices()),
                                 build_vec_idx<size_t>(op.limit_indices()),
                                 build_vec_idx<size_t>(op.strides())));
}

void PPHloExecutor::execute(mlir::pphlo::DbgPrintOp &op) {
  hal::dbg_print(ctx_, lookupValue(op.operand()));
}

void PPHloExecutor::execute(mlir::pphlo::ClampOp &op) {
  getCurrentFrame()->addValue(op.getResult(),
                              hal::clamp(ctx_, lookupValue(op.min()),
                                         lookupValue(op.operand()),
                                         lookupValue(op.max())));
}

void PPHloExecutor::execute(mlir::pphlo::BitcastConvertOp &op) {
  const auto &in_type =
      op.getOperand().getType().dyn_cast<mlir::RankedTensorType>();
  const auto &out_type =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>();

  // bitcast should not change total #bytes, so if sizeof(in_t) != sizeof(out_t)
  // will result to a shape change, thus it's enough to just ensure in_shape ==
  // out_shape
  PPU_ENFORCE(in_type.getShape() == out_type.getShape(),
              "bitcast with different size is not supported yet");

  getCurrentFrame()->addValue(
      op.getResult(),
      hal::bitcast(ctx_, lookupValue(op.getOperand()),
                   type_tools_.isFxpType(out_type) ? DT_FXP : DT_INT,
                   op.elsize()));
}

size_t PPHloExecutor::extractShiftBits(const hal::Value &v) const {
  PPU_ENFORCE(v.is_int());
  const auto arr = hal::dump_public(ctx_, v);
  return arr.at<uint64_t>({});
}

void PPHloExecutor::execute(mlir::pphlo::ShiftRightLogicalOp &op) {
  auto rhs = lookupValue(op.rhs());

  PPU_ENFORCE(rhs.is_public(), "shift bit value needs to be a public");

  std::vector<int64_t> indicies(rhs.shape().size(), 0);

  const auto &lhs = lookupValue(op.lhs());
  auto result = hal::makeValue(lhs, lhs.dtype());

  do {
    auto bits = extractShiftBits(rhs.GetElementAt(indicies));

    result.CopyElementFrom(
        hal::right_shift_logical(ctx_, lhs.GetElementAt(indicies), bits), {},
        indicies);

  } while (bumpIndices<int64_t>(rhs.shape(), absl::MakeSpan(indicies)));

  getCurrentFrame()->addValue(op.getResult(), result);
}

void PPHloExecutor::execute(mlir::pphlo::ShiftLeftOp &op) {
  auto rhs = lookupValue(op.rhs());

  PPU_ENFORCE(rhs.is_public(), "shift bit value needs to be a public");

  std::vector<int64_t> indicies(rhs.shape().size(), 0);

  const auto &lhs = lookupValue(op.lhs());
  auto result = hal::makeValue(lhs, lhs.dtype());

  do {
    auto bits = extractShiftBits(rhs.GetElementAt(indicies));

    result.CopyElementFrom(
        hal::left_sift(ctx_, lhs.GetElementAt(indicies), bits), {}, indicies);

  } while (bumpIndices<int64_t>(rhs.shape(), absl::MakeSpan(indicies)));

  getCurrentFrame()->addValue(op.getResult(), result);
}

void PPHloExecutor::errorUnknownOp(mlir::Operation &op) {
  // These lines of code in theory should not hit.
  // If hit, make a proper error message.
  std::string err_str;
  llvm::raw_string_ostream err(err_str);
  op.print(err);
  PPU_THROW("Unhandled mlir op {}", err.str());
}

void PPHloExecutor::execute(mlir::pphlo::ReverseOp &op) {
  getCurrentFrame()->addValue(
      op.getResult(), hal::reverse(ctx_, lookupValue(op.getOperand()),
                                   build_vec_idx<size_t>(op.dimensions())));
}

void PPHloExecutor::execute(mlir::pphlo::PadOp &op) {
  const auto &operand = lookupValue(op.operand());
  const size_t operand_rank = operand.shape().size();
  const auto &padding_value = lookupValue(op.padding_value());
  PPU_ENFORCE(padding_value.shape().empty());

  auto edge_padding_low = build_vec_idx<size_t>(op.edge_padding_low());
  PPU_ENFORCE(edge_padding_low.size() == operand_rank);
  auto edge_padding_high = build_vec_idx<size_t>(op.edge_padding_high());
  PPU_ENFORCE(edge_padding_high.size() == operand_rank);
  auto interior_padding = build_vec_idx<size_t>(op.interior_padding());
  PPU_ENFORCE(interior_padding.size() == operand_rank);

  getCurrentFrame()->addValue(
      op.getResult(), hal::pad(ctx_, operand, padding_value, edge_padding_low,
                               edge_padding_high, interior_padding));
}

// For one particular placement of a window in a base shape (the placement is
// represented as `window_count_index`), iterates inside the window.
// Translates the window index into base index. If the base index is within
// bound, call `f` with the base index.
static void IterateThroughWindow(
    absl::Span<int64_t> window_shape, absl::Span<int64_t> window_strides,
    absl::Span<int64_t> window_dilation,
    absl::Span<std::pair<int64_t, int64_t>> window_padding,
    absl::Span<int64_t> base_shape, absl::Span<int64_t> base_dilation,
    const absl::Span<int64_t> window_count_index,
    const std::function<void(const std::vector<int64_t> &)> &f) {
  const int64_t rank = base_shape.size();
  std::vector<int64_t> window_index(rank);
  std::fill(window_index.begin(), window_index.end(), 0);
  do {
    std::vector<int64_t> base_index(rank);
    bool out_of_bound = false;
    for (int64_t i = 0; i < rank; ++i) {
      // Padding is applied to the dilated base. Say that padding is 3 and
      // dilation is 2 for some dimension. After applying base dilation and
      // padding, the dimension looks like:
      // P P P E D D E D D ... E D D E P P P
      // where E are the elements and D are the holes. So, the elements are
      // located in indices: padding + k*base_dilation for k = {0, 1, 2, ...}.
      // We are accessing elements in the transformed base at indices:
      // window_count_index * stride + window_index * window_dilation.
      // Solving for k gives us
      // (win_count_i * stride + win_i * win_dilation - pad) / base_dilation
      // When this is a natural number, we index an original element.
      // Otherwise, we index a 0 (pad or hole), and we don't need to apply
      // the callback f.
      base_index[i] = window_count_index[i] * window_strides[i] +
                      window_index[i] * window_dilation[i] -
                      window_padding[i].first;
      if (base_index[i] % base_dilation[i] != 0) {
        out_of_bound = true;
        break;
      }
      base_index[i] /= base_dilation[i];
      if (base_index[i] < 0 || base_index[i] >= base_shape[i]) {
        out_of_bound = true;
        break;
      }
    }
    if (!out_of_bound) {
      f(base_index);
    }
  } while (bumpIndices<int64_t>(window_shape, absl::MakeSpan(window_index)));
}

void PPHloExecutor::execute(mlir::pphlo::ReduceWindowOp &op) {

  PPU_ENFORCE(op->getNumResults() == 1,
              "Variadic reduce window is not supported yet");

  const auto &input = lookupValue(op.inputs());
  const auto &init_val = lookupValue(op.init_values());

  auto window_shape = build_vec_idx<int64_t>(op.window_dimensions());

  // build strides
  std::vector<int64_t> window_strides(window_shape.size(), 1);
  if (op.window_strides().hasValue()) {
    window_strides = build_vec_idx<int64_t>(*op.window_strides());
  }

  // window dilation
  std::vector<int64_t> window_dilations(window_shape.size(), 1);
  if (op.window_dilations().hasValue()) {
    window_dilations = build_vec_idx<int64_t>(*op.window_dilations());
  }

  // window padding
  std::vector<std::pair<int64_t, int64_t>> window_padding(window_shape.size(),
                                                          {0, 0});
  if (op.padding().hasValue()) {
    const auto v = *op.padding();

    PPU_ENFORCE(window_padding.size() * 2 == (size_t)v.size());

    for (size_t idx = 0; idx < window_padding.size(); ++idx) {
      window_padding[idx] = {*(v.getValues<int64_t>().begin() + 2 * idx),
                             *(v.getValues<int64_t>().begin() + 2 * idx + 1)};
    }
  }

  std::vector<int64_t> base_shape;
  for (auto d : input.shape()) {
    base_shape.emplace_back(d);
  }

  // base dilation
  std::vector<int64_t> base_dilation(window_shape.size(), 1);
  if (op.base_dilations().hasValue()) {
    base_dilation = build_vec_idx<int64_t>(*op.base_dilations());
  }

  // For each resulting dimension, calculate and assign computed value.
  auto evaluate_impl = [&](absl::Span<int64_t> output_index) -> hal::Value {
    hal::Value computed_result =
        hal::makeValue(init_val.clone(), init_val.dtype());

    IterateThroughWindow(
        absl::MakeSpan(window_shape), absl::MakeSpan(window_strides),
        absl::MakeSpan(window_dilations), absl::MakeSpan(window_padding),
        absl::MakeSpan(base_shape), absl::MakeSpan(base_dilation), output_index,
        [&](absl::Span<const int64_t> operand_index) {
          auto slice = input.GetElementAt(operand_index);
          PPU_ENFORCE(slice.dtype() != DT_INVALID);
          computed_result =
              executeRegion(op.body(), {computed_result, slice})[0];
        });
    return computed_result;
  };

  // Preallocate result
  auto ret_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  hal::Value ret = hal::broadcast_to(
      ctx_,
      hal::make_value(ctx_, input.vtype(),
                      input.is_int() ? PtBufferView(0) : PtBufferView(0.0F)),
      build_shape(ret_shape));

  // For each window index
  auto old_trace = config_.enable_pphlo_trace;
  config_.enable_pphlo_trace = false;
  std::vector<int64_t> output_index(ret_shape.size(), 0);
  do {
    auto r = evaluate_impl(absl::MakeSpan(output_index));
    std::vector<int64_t> in_idx(r.shape().size(), 0);
    ret.CopyElementFrom(r, absl::MakeSpan(in_idx), output_index);
  } while (
      bumpIndices(absl::MakeSpan(ret_shape), absl::MakeSpan(output_index)));

  config_.enable_pphlo_trace = old_trace;

  getCurrentFrame()->addValue(op.getResult(), std::move(ret));
}

// This is ported from
// https://github.com/tensorflow/tensorflow/blob/bf4c6ad46dac1f7f69911e2bfc48e141a39b40af/tensorflow/compiler/xla/service/hlo_evaluator.cc#L1774
void PPHloExecutor::execute(mlir::pphlo::GatherOp &op) {
  // If input is empty, short circuit
  const auto &operand = lookupValue(op.operand());
  if (operand.numel() == 0) {
    getCurrentFrame()->addValue(op.getResult(), operand);
  }

  const auto &output_shape =
      op.getResult().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  const mlir::pphlo::GatherDimensionNumbersAttr &dim_numbers =
      op.dimension_numbers();

  const auto &start_indices_value = reshapedGatherIndices(
      ctx_, dim_numbers.getIndexVectorDim(), lookupValue(op.start_indices()));

  PPU_ENFORCE(start_indices_value.is_public() && start_indices_value.is_int(),
              "GatherOp start indices must be public integer.");

  auto start_induces =
      hal::test::dump_public_as<int64_t>(ctx_, start_indices_value);

  // We iterate over the gather dimensions in the output shape in an outer
  // loop nest, and iterate over the window dimensions in the output shape in
  // an inner loop nest.
  IndexIterationSpace start_indices_iteration_space =
      iterationSpaceForOutputBatchIndices(output_shape, dim_numbers);
  IndexIterationSpace offset_indices_iteration_space =
      iterationSpaceForOutputOffsetIndices(output_shape.size(),
                                           op.slice_sizes(), dim_numbers);

  // Scratch buffers that hold an index in the output shape and the
  // corresponding index in the input shape.
  // If input is empty, short circuit it
  auto operand_shape =
      op.operand().getType().dyn_cast<mlir::RankedTensorType>().getShape();
  std::vector<int64_t> input_index(operand_shape.size());
  std::vector<int64_t> output_index(output_shape.size());
  std::vector<int64_t> input_index_clamped(operand_shape.size());

  OutputBatchIndexToInputIndex output_batch_index_to_input_index(
      dim_numbers, /*input_shape=*/operand_shape,
      /*output_shape=*/output_shape, start_induces);
  OutputOffsetIndexToInputIndex output_offset_index_to_input_index(
      dim_numbers, /*input_shape=*/operand_shape,
      /*output_shape=*/output_shape);

  hal::Value result = hal::broadcast_to(
      ctx_,
      hal::make_value(ctx_, operand.vtype(),
                      operand.is_int() ? PtBufferView(0) : PtBufferView(0.0F)),
      build_shape(output_shape));

  auto gather_inner_loop_body =
      [&](llvm::ArrayRef<int64_t> output_window_index,
          llvm::ArrayRef<int64_t> input_gather_index,
          llvm::ArrayRef<int64_t> output_gather_index) -> bool {
    llvm::ArrayRef<int64_t> input_window_index =
        output_offset_index_to_input_index(output_window_index);
    for (int i = 0, e = output_index.size(); i < e; i++) {
      output_index[i] = output_gather_index[i] + output_window_index[i];
    }
    for (int i = 0, e = input_gather_index.size(); i < e; i++) {
      int64_t output_dim =
          output_offset_index_to_input_index.input_dim_value_to_output_index(i);
      // If 'output_dim' is -1, it means 'i' is an elided window dim. This
      // means we set the iteration index to 0, so for the purpose of the
      // following calculations we can consider the output dimension size to
      // be 1.
      int64_t output_dim_size = output_dim == -1 ? 1 : output_shape[output_dim];
      // Clamp the gather index so that the gather region fits in the operand.
      // input_index_clamped[i] = clamp(input_gather_index[i], 0,
      //                                       operand_shape.dimensions(i) -
      //                                       output_dim_size);
      input_index_clamped[i] =
          std::min(operand_shape[i] - output_dim_size,
                   std::max(int64_t{0}, input_gather_index[i]));
    }
    for (int i = 0, e = input_index.size(); i < e; i++) {
      input_index[i] = input_index_clamped[i] + input_window_index[i];
    }

    result.CopyElementFrom(operand, input_index, output_index);
    return true;
  };

  auto gather_outer_loop_body =
      [&](llvm::ArrayRef<int64_t> output_gather_index) -> bool {
    llvm::ArrayRef<int64_t> input_gather_index =
        output_batch_index_to_input_index(output_gather_index);
    forEachIndex(output_shape, offset_indices_iteration_space.index_base,
                 offset_indices_iteration_space.index_count,
                 offset_indices_iteration_space.index_incr,
                 [&](llvm::ArrayRef<int64_t> output_window_index) {
                   return gather_inner_loop_body(output_window_index,
                                                 input_gather_index,
                                                 output_gather_index);
                 });
    return true;
  };

  forEachIndex(output_shape, start_indices_iteration_space.index_base,
               start_indices_iteration_space.index_count,
               start_indices_iteration_space.index_incr,
               gather_outer_loop_body);

  getCurrentFrame()->addValue(op.getResult(), std::move(result));
}

// TODO(junfeng): remove restrictions and make it to a general conv2D.
// NOTE(junfeng): Current restrictions:
// 1. dilation must be 1.
// 2. no feature_group_count or batch_group_count.
// 3. 2D convolution only.
// 4. input dimensions must be 4 - b, 0, 1, f
// 5. kernel dimensions must be 4 - 0, 1, i, o
// 6. output dimensions must be 4 - b, 0, 1, o
// Some restrictions could be removed easily while some are not.
void PPHloExecutor::execute(mlir::pphlo::ConvOp &op) {
  // Restriction 1.
  if (op.lhs_dilation().hasValue()) {
    const auto lhs_dilation =
        build_vec_idx<size_t>(op.lhs_dilation().getValue());
    PPU_ENFORCE(std::all_of(lhs_dilation.begin(), lhs_dilation.end(),
                            [](size_t i) { return i == 1; }));
  }
  if (op.rhs_dilation().hasValue()) {
    const auto rhs_dilation =
        build_vec_idx<size_t>(op.rhs_dilation().getValue());
    PPU_ENFORCE(std::all_of(rhs_dilation.begin(), rhs_dilation.end(),
                            [](size_t i) { return i == 1; }));
  }

  // Restriction 2.
  PPU_ENFORCE(op.feature_group_count() == 1);
  PPU_ENFORCE(op.batch_group_count() == 1);

  std::vector<size_t> window_strides =
      op.window_strides().hasValue()
          ? build_vec_idx<size_t>(op.window_strides().getValue())
          : std::vector<size_t>(1, 2);

  std::vector<std::pair<size_t, size_t>> padding(2, {0, 0});

  if (op.padding().hasValue()) {
    padding.clear();
    auto val = op.padding().getValue();
    for (auto iter = val.begin(); iter != val.end(); iter += 2) {
      padding.emplace_back((*iter).getLimitedValue(),
                           (*(iter + 1)).getLimitedValue());
    }
  }

  const auto &dimension_numbers = op.dimension_numbers();

  // Transpose lhs to b01f
  const auto input_batch_dimension = dimension_numbers.getInputBatchDimension();
  const auto input_spatial_dimensions =
      dimension_numbers.getInputSpatialDimensions();
  const auto input_feature_dimension =
      dimension_numbers.getInputFeatureDimension();
  PPU_ENFORCE(input_spatial_dimensions.size() == 2);

  auto input = lookupValue(op.lhs());
  if (input_batch_dimension != 0 || input_spatial_dimensions[0] != 1 ||
      input_spatial_dimensions[1] != 2 || input_feature_dimension != 3) {
    std::vector<size_t> permutation(4);
    permutation[0] = input_batch_dimension;
    permutation[1] = input_spatial_dimensions[0];
    permutation[2] = input_spatial_dimensions[1];
    permutation[3] = input_feature_dimension;
    input = hal::transpose(ctx_, input, permutation);
  }

  // Transpose kernel to 01io
  const auto kernel_spatial_dimensions =
      dimension_numbers.getKernelSpatialDimensions();
  const auto kernel_input_feature_dimension =
      dimension_numbers.getKernelInputFeatureDimension();
  const auto kernel_output_feature_dimension =
      dimension_numbers.getKernelOutputFeatureDimension();
  PPU_ENFORCE(kernel_spatial_dimensions.size() == 2);

  auto kernel = lookupValue(op.rhs());
  if (kernel_spatial_dimensions[0] != 0 || kernel_spatial_dimensions[1] != 1 ||
      kernel_input_feature_dimension != 2 ||
      kernel_output_feature_dimension != 3) {
    std::vector<size_t> permutation(4);
    permutation[0] = kernel_spatial_dimensions[0];
    permutation[1] = kernel_spatial_dimensions[1];
    permutation[2] = kernel_input_feature_dimension;
    permutation[3] = kernel_output_feature_dimension;
    kernel = hal::transpose(ctx_, kernel, permutation);
  }

  // Do conv
  auto ret =
      hal::conv2d_b01f_01io_b01f(ctx_, input, kernel, window_strides, padding);

  // Transpose output from b01f
  const auto output_batch_dimension =
      dimension_numbers.getOutputBatchDimension();
  const auto output_spatial_dimensions =
      dimension_numbers.getOutputSpatialDimensions();
  const auto output_feature_dimension =
      dimension_numbers.getOutputFeatureDimension();
  PPU_ENFORCE(output_spatial_dimensions.size() == 2);

  if (output_batch_dimension != 0 || output_spatial_dimensions[0] != 1 ||
      output_spatial_dimensions[1] != 2 || output_feature_dimension != 3) {
    std::vector<size_t> permutation(4);
    permutation[output_batch_dimension] = 0;
    permutation[output_spatial_dimensions[0]] = 1;
    permutation[output_spatial_dimensions[1]] = 2;
    permutation[output_feature_dimension] = 3;
    ret = hal::transpose(ctx_, ret, permutation);
  }

  getCurrentFrame()->addValue(op.getResult(), std::move(ret));
}

void PPHloExecutor::execute(mlir::pphlo::SortOp &op) {
  // First inputs vectors
  std::vector<hal::Value> inputs;
  inputs.reserve(op->getNumOperands());

  // Prepare inputs
  for (const auto operand : op->getOperands()) {
    inputs.emplace_back(lookupValue(operand));
  }

  const auto res =
      hal::sort(ctx_, inputs, op.dimension(), op.is_stable(), op.is_less());

  // Assign output
  for (const auto &ret : llvm::enumerate(op->getResults())) {
    getCurrentFrame()->addValue(ret.value(), res[ret.index()]);
  }
}

std::vector<hal::Value> PPHloExecutor::executeTerminator(mlir::Operation &op) {
  if (llvm::isa<mlir::ReturnOp>(op) || llvm::isa<mlir::pphlo::ReturnOp>(op)) {
    std::vector<hal::Value> results;
    results.reserve(op.getNumOperands());
    for (const auto operand : op.getOperands()) {
      results.emplace_back(lookupValue(operand));
    }
    return results;
  }
  llvm_unreachable("Unknown block terminator");
}

std::vector<hal::Value> PPHloExecutor::executeBlock(mlir::Block &block) {
  for (auto &op : block.without_terminator()) {
    dispatchOp<
#define GET_OP_LIST
#include "ppu/dialect/pphlo_ops.cc.inc"
        >(op);
  }

  if (auto *termOp = block.getTerminator()) {
    if (config_.enable_pphlo_trace) {
      debug_print(*termOp, true);
    }
    return executeTerminator(*termOp);
  }

  // No terminator
  return {};
}

std::vector<hal::Value>
PPHloExecutor::executeRegion(mlir::Region &region,
                             llvm::ArrayRef<hal::Value> inputs) {
  Frame frame(config_.enable_type_checker);
  for (const auto &blkarg : region.getArguments()) {
    frame.addValue(blkarg, inputs[blkarg.getArgNumber()]);
  }
  frames_.emplace_back(&frame);
  auto results = executeBlock(region.front());
  frames_.pop_back();
  return results;
}

std::vector<hal::Value>
PPHloExecutor::executeFunc(mlir::FuncOp &fcn,
                           llvm::ArrayRef<hal::Value> inputs) {
  return executeRegion(fcn.body(), inputs);
}

std::vector<hal::Value>
PPHloExecutor::executeModule(mlir::ModuleOp &op,
                             llvm::ArrayRef<hal::Value> inputs) {
  // Start from entry function
  auto entry_function = op.lookupSymbol<mlir::FuncOp>("main");
  PPU_ENFORCE(entry_function);

  return executeFunc(entry_function, inputs);
}

void PPHloExecutor::debug_print(mlir::Operation &op, bool before) const {
  if (before) {
    if (ctx_->lctx() && ctx_->lctx()->Rank() == 0) {
      std::string buf;
      llvm::raw_string_ostream debug_stream(buf);
      op.print(debug_stream);
      SPDLOG_INFO("PPHLO {}", debug_stream.str());
    }
  } else {
    for (const auto &ret : llvm::enumerate(op.getResults())) {
      if (ctx_->lctx() && ctx_->lctx()->Rank() == 0) {
        SPDLOG_INFO("PPHLO ret {}", ret.index());
      }
      hal::dbg_print(ctx_, getCurrentFrame()->getValue(ret.value()));
    }
  }
}

} // namespace ppu::device
