import numpy as np
from os import PathLike
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional, Type, Union
from harp.model import Model

datapath = Path(__file__).parent


@dataclass
class DataFileParam:
    path: Union[str, PathLike]
    expected_rows: int
    expected_cols: Optional[Iterable[str]] = None
    expected_address: Optional[int] = None
    expected_dtype: Optional[np.dtype] = None
    expected_length: Optional[int] = None
    expected_error: Optional[Type[BaseException]] = None
    keep_type: bool = False

    def __post_init__(self):
        self.path = datapath / self.path


@dataclass
class DeviceSchemaParam:
    path: Union[str, PathLike]
    expected_whoAmI: int
    expected_device: Optional[int] = None
    expected_registers: Optional[Iterable[str]] = None
    expected_error: Optional[Type[BaseException]] = None

    def __post_init__(self):
        self.path = datapath / self.path

    def assert_schema(self, device: Model):
        assert device.whoAmI == self.expected_whoAmI
        if self.expected_registers:
            for register in self.expected_registers:
                assert register in device.registers
