import pytest
from os import PathLike
from typing import Iterable, Optional, Type, Union
from pytest import mark
from pathlib import Path
from dataclasses import dataclass
from harp.schema import read_schema

datapath = Path(__file__).parent


@dataclass
class DeviceSchemaParam:
    path: Union[str, PathLike]
    expected_whoAmI: int
    expected_device: Optional[int] = None
    expected_registers: Optional[Iterable[str]] = None
    expected_error: Optional[Type[BaseException]] = None

    def __post_init__(self):
        self.path = datapath / self.path


testdata = [
    DeviceSchemaParam(
        path="data/device.yml",
        expected_whoAmI=0,
        expected_registers=["DigitalInputMode"],
    )
]


@mark.parametrize("schemaFile", testdata)
def test_read_schema(schemaFile: DeviceSchemaParam):
    schema = read_schema(schemaFile.path)
    assert schema.whoAmI == schemaFile.expected_whoAmI
    if schemaFile.expected_registers:
        for register in schemaFile.expected_registers:
            assert register in schema.registers


if __name__ == "__main__":
    pytest.main()
