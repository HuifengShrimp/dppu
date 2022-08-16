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


// clang-format off
// To run the example, start two terminals:
// > bazel run //examples/cpp:simple_lr -- --dataset=examples/cpp/data/perfect_logit_a.csv --has_label=true
// > bazel run //examples/cpp:simple_lr -- --dataset=examples/cpp/data/perfect_logit_b.csv --rank=1
// clang-format on

#include <fstream>
#include <iostream>
#include <vector>

#include "examples/cpp/utils.h"
#include "spdlog/spdlog.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xcsv.hpp"
#include "xtensor/xio.hpp"

#include "ppu/device/colocated_io.h"
#include "ppu/hal/type_cast.h"
#include "ppu/numpy/numpy.h"

namespace np = ppu::numpy;

ppu::hal::Value train_step(ppu::HalContext* ctx, const ppu::hal::Value& x,
                           const ppu::hal::Value& y, const ppu::hal::Value& w) {
  // // Padding x
  // auto padding = np::ones(ctx, {x.shape()[0], int64_t(1)}, ppu::DT_FXP);
  // auto padded_x = np::concatenate(ctx, x, ppu::hal::p2s(ctx, padding), 1);
  // auto pred = np::logistic(ctx, np::matmul(ctx, padded_x, w));

  // SPDLOG_TRACE("[FSS-LR] Err = Pred - Y");
  // auto err = np::sub(ctx, pred, y);

  // SPDLOG_TRACE("[FSS-LR] Grad = X.t * Err");
  // auto grad = np::matmul(ctx, np::transpose(ctx, padded_x), err);

  // SPDLOG_TRACE("[FSS-LR] Step = LR / B * Grad");
  // auto lr = np::make_public(ctx, 0.0001F);
  // auto msize = np::make_public(ctx, static_cast<float>(y.shape()[0]));
  // auto p1 = np::mul(ctx, lr, np::reciprocal(ctx, msize));
  // auto step = np::mul(ctx, p1, grad);

  size_t m = x.elsize(), n = 10;
  // std::cout<<"*********train_step**************"<<std::endl;
  // std::cout<<"*********************************"<<std::endl;
  SPDLOG_TRACE("[FSS-LR] ");
  auto new_w = np::logreg(ctx, x, w, y, m, n);

  return new_w;
}

ppu::hal::Value train(ppu::HalContext* ctx, const ppu::hal::Value& x,
                      const ppu::hal::Value& y, size_t num_epoch,
                      size_t bsize) {
  const size_t num_iter = x.shape()[0] / bsize;
  auto w = np::zeros(ctx, {x.shape().at(1) + 1, 1}, ppu::DT_FXP);

  // Run train loop
  for (size_t epoch = 0; epoch < num_epoch; ++epoch) {
    for (size_t iter = 0; iter < num_iter; ++iter) {
      SPDLOG_INFO("Running train iteration {}", iter);

      const auto rows_beg = iter * bsize;
      const auto rows_end = rows_beg + bsize;

      // std::cout<<"**********bsize**************"<<std::endl;
      // std::cout<<"*********"<<bsize<<"*********"<<std::endl;
      // std::cout<<"*****************************"<<std::endl;

      const auto x_slice =
          np::slice(ctx, x, {rows_beg, 0},
                    {rows_end, static_cast<size_t>(x.shape()[1])}, {});

      const auto y_slice =
          np::slice(ctx, y, {rows_beg, 0},
                    {rows_end, static_cast<size_t>(y.shape()[1])}, {});


      w = train_step(ctx, x_slice, y_slice, w);
    }
  }

  return w;
}

ppu::hal::Value inference(ppu::HalContext* ctx, const ppu::hal::Value& x,
                          const ppu::hal::Value& weight) {
  auto padding = np::ones(ctx, {x.shape()[0], int64_t(1)}, ppu::DT_FXP);
  auto padded_x = np::concatenate(ctx, x, ppu::hal::p2s(ctx, padding), 1);
  return np::matmul(ctx, padded_x, weight);
}

float SSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  float sse = 0;

  for (auto y_true_iter = y_true.begin(), y_pred_iter = y_pred.begin();
       y_true_iter != y_true.end() && y_pred_iter != y_pred.end();
       ++y_pred_iter, ++y_true_iter) {
    sse += std::pow(*y_true_iter - *y_pred_iter, 2);
  }
  return sse;
}

float MSE(const xt::xarray<float>& y_true, const xt::xarray<float>& y_pred) {
  auto sse = SSE(y_true, y_pred);

  return sse / static_cast<float>(y_true.size());
}

llvm::cl::opt<std::string> Dataset("dataset", llvm::cl::init("data.csv"),
                                   llvm::cl::desc("only csv is supported"));
llvm::cl::opt<uint32_t> SkipRows(
    "skip_rows", llvm::cl::init(1),
    llvm::cl::desc("skip number of rows from dataset"));
llvm::cl::opt<bool> HasLabel(
    "has_label", llvm::cl::init(false),
    llvm::cl::desc("if true, label is the last column of dataset"));
llvm::cl::opt<uint32_t> BatchSize("batch_size", llvm::cl::init(21),
                                  llvm::cl::desc("size of each batch"));
llvm::cl::opt<uint32_t> NumEpoch("num_epoch", llvm::cl::init(1),
                                 llvm::cl::desc("number of epoch"));

std::pair<ppu::hal::Value, ppu::hal::Value> infeed(ppu::device::Processor* proc,
                                                   const xt::xarray<float>& ds,
                                                   bool self_has_label) {
  auto hctx = proc->hctx();
  ppu::device::ColocatedIo io(proc);
  if (self_has_label) {
    // the last column is label.
    using namespace xt::placeholders;  // required for `_` to work
    xt::xarray<float> dx =
        xt::view(ds, xt::all(), xt::range(_, ds.shape(1) - 1));
    xt::xarray<float> dy =
        xt::view(ds, xt::all(), xt::range(ds.shape(1) - 1, _));
    io.setVar(fmt::format("x-{}", hctx->lctx()->Rank()), dx);
    io.setVar("label", dy);
  } else {
    io.setVar(fmt::format("x-{}", hctx->lctx()->Rank()), ds);
  }
  io.sync();

  auto x = io.getVar("x-0");
  // Concatnate all slices
  for (size_t idx = 1; idx < io.world_size(); ++idx) {
    x = ppu::numpy::concatenate(hctx, x, io.getVar(fmt::format("x-{}", idx)),
                                1);
  }
  auto y = io.getVar("label");

  return std::make_pair(x, y);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  // read dataset.
  xt::xarray<float> ds;
  {
    std::ifstream file(Dataset.getValue());
    if (!file) {
      spdlog::error("open file={} failed", Dataset.getValue());
      exit(-1);
    }
    ds = xt::load_csv<float>(file, ',', SkipRows.getValue());
  }

  auto proc = MakeProcessor();
  auto hctx = proc->hctx();

  const auto& [x, y] = infeed(proc.get(), ds, HasLabel.getValue());

  const auto w = train(hctx, x, y, NumEpoch.getValue(), BatchSize.getValue());

  const auto scores = inference(hctx, x, w);

  xt::xarray<float> revealed_labels =
      np::dump_xarray<float>(hctx, np::reveal(hctx, y));
  xt::xarray<float> revealed_scores =
      np::dump_xarray<float>(hctx, np::reveal(hctx, scores));

  auto mse = MSE(revealed_labels, revealed_scores);
  std::cout << "MSE = " << mse << "\n";

  return 0;
}
