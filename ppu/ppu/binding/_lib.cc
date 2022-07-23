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


#include "fmt/format.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "ppu/compiler/common/compilation_context.h"
#include "ppu/compiler/compile.h"
#include "ppu/core/encoding.h"
#include "ppu/device/io_accessor.h"
#include "ppu/device/processor.h"
#include "ppu/link/link.h"
#include "ppu/psi/core/ecdh_psi.h"
#include "ppu/psi/psi.h"

namespace py = pybind11;

namespace ppu {

#define NO_GIL py::call_guard<py::gil_scoped_release>()

// TODO(jint) research, what happened when return shared_ptr by reference.
#define RVP_COPY py::return_value_policy::copy

void BindLink(py::module& m) {
  using link::Context;
  using link::ContextDesc;

  // TODO(jint) expose this tag to python?
  constexpr char PY_CALL_TAG[] = "PY_CALL";

  m.doc() = R"pbdoc(
              PPU Link Library
                  )pbdoc";

  py::class_<ContextDesc::Party>(
      m, "Party", "The party that participate the secure computation")
      .def_readonly("id", &ContextDesc::Party::id, "the id, unique per link")
      .def_readonly("host", &ContextDesc::Party::host, "host address")
      .def("__repr__", [](const ContextDesc::Party& self) {
        return fmt::format("Party(id={}, host={})", self.id, self.host);
      });

  py::class_<ContextDesc>(
      m, "Desc", "Link description, describes parties which joins the link")
      .def(py::init<>())
      .def_readwrite("id", &ContextDesc::id, "the uuid")
      .def_readonly("parties", &ContextDesc::parties,
                    "the parties that joins the computation")
      .def_readwrite("connect_retry_times", &ContextDesc::connect_retry_times)
      .def_readwrite("connect_retry_interval_ms",
                     &ContextDesc::connect_retry_interval_ms)
      .def_readwrite("recv_timeout_ms", &ContextDesc::recv_timeout_ms)
      .def_readwrite("http_max_payload_size",
                     &ContextDesc::http_max_payload_size)
      .def_readwrite("http_timeout_ms", &ContextDesc::http_timeout_ms)
      .def_readwrite("brpc_channel_protocol",
                     &ContextDesc::brpc_channel_protocol)
      .def_readwrite("brpc_channel_connection_type",
                     &ContextDesc::brpc_channel_connection_type)
      .def(
          "add_party",
          [](ContextDesc& desc, std::string id, std::string host) {
            desc.parties.push_back({id, host});
          },
          "add a party to the link");

  // expose shared_ptr<Context> to py
  py::class_<Context, std::shared_ptr<Context>>(m, "Context", "the link handle")
      .def("__repr__",
           [](const Context* self) {
             return fmt::format("Link(id={}, rank={}/{})", self->Id(),
                                self->Rank(), self->WorldSize());
           })
      .def(
          "id", [](const Context* self) { return self->Id(); },
          "the unique link id")
      .def_property_readonly(
          "rank", [](const Context* self) { return self->Rank(); },
          py::return_value_policy::copy, "my rank of the link")
      .def_property_readonly(
          "world_size", [](const Context* self) { return self->WorldSize(); },
          py::return_value_policy::copy, "the number of parties")
      .def(
          "spawn",
          [](const std::shared_ptr<Context>& self) {
            return std::shared_ptr<Context>(self->Spawn());
          },
          NO_GIL, "spawn a sub-link, advanced skill")
      .def(
          "barrier",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self) -> void {
            return link::Barrier(self, PY_CALL_TAG);
          },
          NO_GIL,
          "Blocks until all parties have reached this routine, aka MPI_Barrier")
      .def(
          "all_gather",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::string& in) -> std::vector<std::string> {
            // FIXME: Find a way to steal memory here...
            auto bufs = link::AllGather(
                self, {in.c_str(), static_cast<int64_t>(in.size())},
                PY_CALL_TAG);
            std::vector<std::string> ret(bufs.size());
            for (size_t idx = 0; idx < bufs.size(); ++idx) {
              ret[idx] = std::string(bufs[idx].data<char>(), bufs[idx].size());
            }
            return ret;
          },
          NO_GIL,
          "Gathers data from all parties and distribute the combined data to "
          "all parties, aka MPI_Allgather")
      .def(
          "gather",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::string& in,
                         size_t root) -> std::vector<std::string> {
            // FIXME: Find a way to steal memory here...
            auto bufs = link::Gather(
                self, {in.c_str(), static_cast<int64_t>(in.size())}, root,
                PY_CALL_TAG);
            std::vector<std::string> ret(bufs.size());
            for (size_t idx = 0; idx < bufs.size(); ++idx) {
              ret[idx] = std::string(bufs[idx].data<char>(), bufs[idx].size());
            }
            return ret;
          },
          NO_GIL, "Gathers values from other parties, aka MPI_Gather")
      .def(
          "broadcast",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::string& in, size_t root) -> std::string {
            // FIXME: Find a way to steal memory here...
            auto buf = link::Broadcast(
                self, {in.c_str(), static_cast<int64_t>(in.size())}, root,
                PY_CALL_TAG);
            return {buf.data<char>(), static_cast<size_t>(buf.size())};
          },
          NO_GIL,
          "Broadcasts a message from the party with rank 'root' to all other "
          "parties, aka MPI_Bcast")
      .def(
          "scatter",
          [&PY_CALL_TAG](const std::shared_ptr<Context>& self,
                         const std::vector<std::string>& in,
                         size_t root) -> std::string {
            // FIXME: Find a way to steal memory here...
            std::vector<Buffer> bufs(in.size());
            for (size_t idx = 0; idx < bufs.size(); ++idx) {
              bufs[idx] = Buffer(in[idx].c_str(), in[idx].size());
            }
            auto buf = link::Scatter(self, bufs, root, PY_CALL_TAG);
            return {buf.data<char>(), static_cast<size_t>(buf.size())};
          },
          NO_GIL,
          "Sends data from one party to all other parties, aka MPI_Scatter");

  m.def("create_brpc",
        [](const ContextDesc& desc,
           size_t self_rank) -> std::shared_ptr<Context> {
          py::gil_scoped_release release;
          auto ctx = link::FactoryBrpc().CreateContext(desc, self_rank);
          ctx->ConnectToMesh();
          return ctx;
        });

  m.def("create_mem",
        [](const ContextDesc& desc,
           size_t self_rank) -> std::shared_ptr<Context> {
          py::gil_scoped_release release;
          auto ctx = link::FactoryMem().CreateContext(desc, self_rank);
          ctx->ConnectToMesh();
          return ctx;
        });
}

// Wrap Processor, it's workaround for protobuf pybind11/protoc conflict.
class RuntimeWrapper {
  std::unique_ptr<ppu::device::Processor> processor_;

 public:
  explicit RuntimeWrapper(std::shared_ptr<link::Context> lctx,
                          const std::string& config_pb) {
    ppu::RuntimeConfig config;
    PPU_ENFORCE(config.ParseFromString(config_pb));

    processor_ = std::make_unique<ppu::device::Processor>(config, lctx);
  }

  void Run(const py::bytes& exec_pb) {
    ppu::ExecutableProto exec;
    PPU_ENFORCE(exec.ParseFromString(exec_pb));
    return processor_->run(exec);
  }

  void SetVar(const std::string& name, const py::bytes& value) {
    return processor_->setVar(name, value);
  }

  py::bytes GetVar(const std::string& name) const {
    return py::bytes(processor_->getVar(name));
  }
};

#define FOR_PY_FORMATS(FN) \
  FN("b", PT_I8)           \
  FN("h", PT_I16)          \
  FN("i", PT_I32)          \
  FN("l", PT_I64)          \
  FN("q", PT_I64)          \
  FN("B", PT_U8)           \
  FN("H", PT_U16)          \
  FN("I", PT_U32)          \
  FN("L", PT_U64)          \
  FN("Q", PT_U64)          \
  FN("f", PT_F32)          \
  FN("d", PT_F64)          \
  FN("?", PT_BOOL)

// https://docs.python.org/3/library/struct.html#format-characters
// https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
// Note: python and numpy has different type string, here pybind11 uses numpy's
// definition
ppu::PtType PyFormatToPtType(const std::string& format) {
#define CASE(FORMAT, PT_TYPE) \
  if (format == FORMAT) return PT_TYPE;

  if (false) {
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  PPU_THROW("unknown py format={}", format);
}

std::string PtTypeToPyFormat(ppu::PtType pt_type) {
#define CASE(FORMAT, PT_TYPE) \
  if (pt_type == PT_TYPE) return FORMAT;

  if (false) {
  }
  FOR_PY_FORMATS(CASE)

#undef CASE
  PPU_THROW("unknown pt_type={}", pt_type);
}

template <typename Iter>
std::vector<int64_t> ByteToElementStrides(const Iter& begin, const Iter& end,
                                          size_t elsize) {
  std::vector<int64_t> ret(std::distance(begin, end));
  std::transform(begin, end, ret.begin(), [&](int64_t c) -> int64_t {
    PPU_ENFORCE(c % elsize == 0);
    return c / elsize;
  });
  return ret;
}

constexpr void SizeCheck() {
  static_assert(sizeof(intptr_t) == 8, "PPU only supports 64-bit system");
  static_assert(sizeof(long long) == 8, "PPU assumes size of longlong == 8");
  static_assert(sizeof(unsigned long long) == 8,
                "PPU assumes size of ulonglong == 8");
}

class IoWrapper {
  std::unique_ptr<ppu::device::IoAccessor> ptr_;

 public:
  IoWrapper(size_t world_size, const std::string& config_pb) {
    ppu::RuntimeConfig config;
    PPU_ENFORCE(config.ParseFromString(config_pb));

    ptr_ = std::make_unique<ppu::device::IoAccessor>(world_size, config);
  }

  std::vector<py::bytes> MakeShares(const py::array& arr, int visibility) {
    // When working with Python, do a sataic size check, this has no runtime
    // cost
    SizeCheck();

    const py::buffer_info& binfo = arr.request();
    const PtType pt_type = PyFormatToPtType(binfo.format);

    ppu::PtBufferView view(
        binfo.ptr, pt_type,
        std::vector<int64_t>(binfo.shape.begin(), binfo.shape.end()),
        ByteToElementStrides(binfo.strides.begin(), binfo.strides.end(),
                             binfo.itemsize));

    auto vals = ptr_->makeShares(ppu::Visibility(visibility), view);
    std::vector<py::bytes> serialized(vals.size());
    for (size_t idx = 0; idx < vals.size(); ++idx) {
      std::string s;
      PPU_ENFORCE(vals[idx].SerializeToString(&s));
      serialized[idx] = py::bytes(s);
    }

    return serialized;
  }

  py::array reconstruct(const std::vector<std::string>& vals) {
    std::vector<ppu::ValueProto> val_protos;
    PPU_ENFORCE(vals.size() > 0);
    for (size_t idx = 0; idx < vals.size(); ++idx) {
      ppu::ValueProto vp;
      PPU_ENFORCE(vp.ParseFromString(vals[idx]));
      val_protos.push_back(std::move(vp));
    }

    // sanity
    for (size_t idx = 1; idx < vals.size(); ++idx) {
      const auto& cur = val_protos[idx];
      const auto& prev = val_protos[idx - 1];
      PPU_ENFORCE(cur.type_data() == prev.type_data());
    }

    auto type = Type::fromString(val_protos.front().type_data());
    const PtType pt_type = ppu::GetDecodeType(type.as<ValueTy>()->dtype());
    auto buf = ptr_->combineShares(val_protos, pt_type);
    std::vector<size_t> shape = {val_protos.at(0).shape().dims().begin(),
                                 val_protos.at(0).shape().dims().end()};

    return py::array(py::dtype(PtTypeToPyFormat(pt_type)), shape, buf.data());
  }
};

void BindLibs(py::module& m) {
  m.doc() = R"pbdoc(
              PPU Mixed Library
                  )pbdoc";

  m.def(
      "ecdh_psi",
      [](const std::shared_ptr<link::Context>& lctx,
         const std::vector<std::string>& items,
         int64_t rank) -> std::vector<std::string> {
        // Sanity rank
        size_t target_rank = rank;
        if (rank == -1) {
          target_rank = link::kAllRank;
        } else if (rank < -1) {
          PPU_THROW("rank should be >= -1, got {}", rank);
        }
        return psi::RunEcdhPsi(lctx, items, target_rank);
      },
      NO_GIL);

  m.def(
      "ecdh_3pc_psi",
      [](const std::shared_ptr<link::Context>& lctx,
         const std::vector<std::string>& selected_fields,
         const std::string& input_path, const std::string& output_path,
         bool should_sort, psi::PsiReport* report) -> void {
        psi::LegacyPsiOptions psi_opts;
        psi_opts.base_options.link_ctx = lctx;
        psi_opts.base_options.field_names = selected_fields;
        psi_opts.base_options.in_path = input_path;
        psi_opts.base_options.out_path = output_path;
        psi_opts.base_options.should_sort = should_sort;
        psi_opts.psi_protocol = psi::kPsiProtocolEcdh;

        auto executor = psi::BuildPsiExecutor(psi_opts);
        executor->Init();
        executor->Run(report);
      },
      NO_GIL);

  m.def(
      "kkrt_2pc_psi",
      [](const std::shared_ptr<link::Context>& lctx,
         const std::vector<std::string>& selected_fields,
         const std::string& input_path, const std::string& output_path,
         bool should_sort, psi::PsiReport* report) -> void {
        psi::LegacyPsiOptions psi_opts;
        psi_opts.base_options.link_ctx = lctx;
        psi_opts.base_options.field_names = selected_fields;
        psi_opts.base_options.in_path = input_path;
        psi_opts.base_options.out_path = output_path;
        psi_opts.base_options.should_sort = should_sort;
        psi_opts.psi_protocol = psi::kPsiProtocolKkrt;

        auto executor = psi::BuildPsiExecutor(psi_opts);
        executor->Init();
        executor->Run(report);
      },
      NO_GIL);

  m.def(
      "ecdh_2pc_psi",
      [](const std::shared_ptr<link::Context>& lctx,
         const std::vector<std::string>& selected_fields,
         const std::string& input_path, const std::string& output_path,
         size_t num_bins, bool should_sort, psi::PsiReport* report) -> void {
        psi::LegacyPsiOptions psi_opts;
        psi_opts.base_options.link_ctx = lctx;
        psi_opts.base_options.field_names = selected_fields;
        psi_opts.base_options.in_path = input_path;
        psi_opts.base_options.out_path = output_path;
        psi_opts.base_options.should_sort = should_sort;
        psi_opts.num_bins = num_bins;
        psi_opts.psi_protocol = psi::kPsiProtocolEcdh2PC;

        auto executor = psi::BuildPsiExecutor(psi_opts);
        executor->Init();
        executor->Run(report);
      },
      NO_GIL);
}

PYBIND11_MODULE(_lib, m) {
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const ppu::Exception& e) {
      // Translate this exception to a standard RuntimeError
      PyErr_SetString(PyExc_RuntimeError,
                      fmt::format("what: \n\t{}\nstacktrace: \n{}\n", e.what(),
                                  e.stack_trace())
                          .c_str());
    }
  });

  // bind ppu virtual machine.
  py::class_<RuntimeWrapper>(m, "RuntimeWrapper", "PPU virtual device")
      .def(py::init<std::shared_ptr<link::Context>, std::string>(), NO_GIL)
      .def("Run", &RuntimeWrapper::Run, NO_GIL)
      .def("SetVar", &RuntimeWrapper::SetVar, NO_GIL)
      .def("GetVar", &RuntimeWrapper::GetVar, NO_GIL);

  // bind ppu io suite.
  py::class_<IoWrapper>(m, "IoWrapper", "PPU VM IO")
      .def(py::init<size_t, std::string>())
      .def("MakeShares", &IoWrapper::MakeShares)
      .def("Reconstruct", &IoWrapper::reconstruct);

  // bind compiler.
  // TODO: use type compile :: IrProto -> IrProto
  m.def(
      "compile",
      [](const py::bytes& hlo_text, const std::string& input_visbility_map,
         const std::string& dump_path) {
        py::scoped_ostream_redirect stream(
            std::cout,                                 // std::ostream&
            py::module_::import("sys").attr("stdout")  // Python output
        );

        ppu::compiler::CompilationContext ctx;
        ctx.setInputVisibilityString(input_visbility_map);

        if (!dump_path.empty()) {
          ctx.enablePrettyPrintWithDir(dump_path);
        }

        return py::bytes(ppu::compiler::compile(&ctx, hlo_text));
      },
      "ppu compile.", py::arg("hlo_text"), py::arg("vis_map"),
      py::arg("dump_path"));

  // bind ppu libs.
  py::module link_m = m.def_submodule("link");
  BindLink(link_m);

  py::module libs_m = m.def_submodule("libs");
  BindLibs(libs_m);

  py::class_<psi::PsiReport>(libs_m, "PsiReport")
      .def(py::init())
      .def_readwrite("intersection_count", &psi::PsiReport::intersection_count)
      .def_readwrite("original_count", &psi::PsiReport::original_count);
}

}  // namespace ppu
