import pytest
import numpy as np
from typing import Iterable, Optional, Type
from contextlib import nullcontext
from pytest import mark
from pathlib import Path
from dataclasses import dataclass
from harp.io import read

datapath = Path(__file__).parent


@dataclass
class DataFileParam:
    path: str
    expected_rows: int
    expected_cols: Optional[Iterable[str]] = None
    expected_address: Optional[int] = None
    expected_dtype: Optional[np.dtype] = None
    expected_length: Optional[int] = None
    expected_error: Optional[Type[BaseException]] = None
    keep_type: bool = False

    def __post_init__(self):
        self.path = datapath / self.path


testdata = [
    DataFileParam(path="data/device_0.bin", expected_rows=1),
    DataFileParam(
        path="data/device_0.bin",
        expected_rows=1,
        expected_address=1,  # actual address is 0
        expected_error=ValueError,
    ),
    DataFileParam(
        path="data/device_0.bin",
        expected_rows=1,
        expected_dtype="uint8",  # actual dtype is uint16
        expected_error=ValueError,
    ),
    DataFileParam(
        path="data/device_0.bin",
        expected_rows=1,
        expected_length=2,  # actual length is 1
        expected_error=ValueError,
    ),
    DataFileParam(path="data/write_0.bin", expected_address=0, expected_rows=4),
    DataFileParam(
        path="data/write_0.bin", expected_address=0, expected_rows=4, keep_type=True
    ),
]


@mark.parametrize("dataFile", testdata)
def test_read(dataFile: DataFileParam):
    context = pytest.raises if dataFile.expected_error is not None else nullcontext
    with context(dataFile.expected_error):
        data = read(
            dataFile.path,
            address=dataFile.expected_address,
            dtype=dataFile.expected_dtype,
            length=dataFile.expected_length,
            keep_type=dataFile.keep_type,
        )
        assert len(data) == dataFile.expected_rows
        if dataFile.keep_type:
            assert "type" in data.columns and data["type"].dtype == "category"

        if dataFile.expected_cols is not None:
            for col in dataFile.expected_cols:
                assert col in data.columns


if __name__ == "__main__":
    pytest.main()
