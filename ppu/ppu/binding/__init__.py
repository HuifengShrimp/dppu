
# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

from ppu.ppu_pb2 import (DataType, Visibility, PtType, ProtocolKind, FieldType,
                         ValueProto, ShapeProto, RuntimeConfig, ExecutableProto,
                         IrProto, IrType)

from . import _lib
from google.protobuf.json_format import MessageToJson


class Runtime(object):
    """ The PPU Virtual Machine Slice.
    """
    def __init__(self, link: _lib.link.Context, config: RuntimeConfig):
        """Constructor of a PPU Runtime.

        Args:
            link (_lib.link.Context): Link context.
            config (RuntimeConfig): PPU Runtime Config.
        """
        self._vm = _lib.RuntimeWrapper(link, config.SerializeToString())

    def run(self, executable: ExecutableProto) -> None:
        """Run a PPU executable.

        Args:
            executable (ExecutableProto): executable.

        """
        return self._vm.Run(executable.SerializeToString())

    def set_var(self, name: str, value: ValueProto) -> None:
        """Set a PPU value.

        Args:
            name (str): Id of value.
            value (ValueProto): value data.

        """
        return self._vm.SetVar(name, value.SerializeToString())

    def get_var(self, name: str) -> ValueProto:
        """Get a PPU value.

        Args:
            name (str): Id of value.

        Returns:
            ValueProto: Data data.
        """
        ret = ValueProto()
        ret.ParseFromString(self._vm.GetVar(name))
        return ret


class Io(object):
    """ The PPU IO interface.
    """
    def __init__(self, world_size: int, config: RuntimeConfig):
        """Constructor of a PPU Io.

        Args:
            world_size (int): # of participants of PPU Device.
            config (RuntimeConfig): PPU Runtime Config.
        """
        self._io = _lib.IoWrapper(world_size, config.SerializeToString())

    def make_shares(self, x: np.ndarray, vtype: Visibility) -> [ValueProto]:
        """Convert from numpy array to list of PPU value(s).

        Args:
            x (np.ndarray): input.
            vtype (Visibility): visibility.

        Returns:
            [ValueProto]: output.
        """
        str_shares = self._io.MakeShares(x, vtype)
        rets = []
        for str_share in str_shares:
            value_share = ValueProto()
            value_share.ParseFromString(str_share)
            rets.append(value_share)
        return rets

    def reconstruct(self, xs: [ValueProto]) -> np.ndarray:
        """Convert from list of PPU value(s) to numpy array.

        Args:
            xs (ValueProto]): input.

        Returns:
            np.ndarray: output.
        """
        str_shares = [x.SerializeToString() for x in xs]
        return self._io.Reconstruct(str_shares)


def compile(src: IrProto) -> IrProto:
    """ Compile from XLA HLO to PPU HLO.
    """
    if src.ir_type == IrType.IR_XLA_HLO:
        mlir = _lib.compile(src.code, MessageToJson(src.meta), "")
        mlir_proto = IrProto()
        mlir_proto.code = mlir
        mlir_proto.ir_type = IrType.IR_MLIR_PPU
        return mlir_proto
    else:
        raise NameError("Unknown ir type")
