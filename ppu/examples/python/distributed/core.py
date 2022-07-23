
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


from abc import abstractmethod
from collections.abc import Callable
from typing import Any


class DeviceObject:
    """
    Reference to a device object.

    TODO: Lifetime management.
    """
    def __init__(self, zrank: int, zname: str):
        self.zrank = zrank
        self.zname = zname

    def __repr__(self):
        return f'ZREF:P{self.zrank}:{self.zname}'


class DeviceClient:
    """
    Represent a device.
    """
    def __init__(self, zctx, zrank: int):
        self.zctx = zctx
        self.zrank = zrank

    @abstractmethod
    def kind(self):
        pass

    @abstractmethod
    def call(self, fn: Callable, *args, **kwargs) -> DeviceObject:
        """ Call a python function on this device
        """
        pass

    @abstractmethod
    def put(self, obj, name) -> DeviceObject:
        """ Put an object to this device
        """
        pass

    @abstractmethod
    def get(self, ref: DeviceObject) -> Any:
        """ Get a device object from this device
        """
        pass
