from contextlib import nullcontext

import numpy as np
import pytest
from pytest import mark

from harp.io import MessageType, read
from tests.params import DataFileParam

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
        expected_dtype=np.dtype("uint8"),  # actual dtype is uint16
        expected_error=ValueError,
    ),
    DataFileParam(
        path="data/device_0.bin",
        expected_rows=1,
        expected_length=2,  # actual length is 1
        expected_error=ValueError,
    ),
    DataFileParam(path="data/write_0.bin", expected_address=0, expected_rows=4),
    DataFileParam(path="data/write_0.bin", expected_address=0, expected_rows=4, keep_type=True),
]


@mark.parametrize("dataFile", testdata)
def test_read(dataFile: DataFileParam):
    context = pytest.raises if dataFile.expected_error else nullcontext
    with context(dataFile.expected_error):  # type: ignore
        data = read(
            dataFile.path,
            address=dataFile.expected_address,
            dtype=dataFile.expected_dtype,
            length=dataFile.expected_length,
            keep_type=dataFile.keep_type,
        )
        assert len(data) == dataFile.expected_rows
        if dataFile.keep_type:
            assert MessageType.__name__ in data.columns and data[MessageType.__name__].dtype == "category"

        if dataFile.expected_cols:
            for col in dataFile.expected_cols:
                assert col in data.columns
