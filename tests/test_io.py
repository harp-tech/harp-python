from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest
from pytest import mark

from harp.io import REFERENCE_EPOCH, MessageType, format, read
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
    DataFileParam(path="data/device_0.bin", expected_rows=300, repeat_data=300),
    DataFileParam(path="data/device_0.bin", expected_rows=1, epoch=REFERENCE_EPOCH),
    DataFileParam(path="data/empty_0.bin", expected_rows=0, epoch=REFERENCE_EPOCH),
    DataFileParam(path="data/empty_0.bin", expected_rows=0),
]


@mark.parametrize("dataFile", testdata)
def test_read(dataFile: DataFileParam):
    context = pytest.raises if dataFile.expected_error else nullcontext
    with context(dataFile.expected_error):  # type: ignore
        file_or_buf = dataFile.path
        if dataFile.repeat_data:
            with open(file_or_buf, "rb") as f:
                file_or_buf = f.read() * dataFile.repeat_data

        data = read(
            file_or_buf,
            address=dataFile.expected_address,
            dtype=dataFile.expected_dtype,
            length=dataFile.expected_length,
            epoch=dataFile.epoch,
            keep_type=dataFile.keep_type,
        )

        assert len(data) == dataFile.expected_rows
        assert isinstance(data.index, pd.DatetimeIndex if dataFile.epoch else pd.Index)
        if dataFile.keep_type:
            assert MessageType.__name__ in data.columns and data[MessageType.__name__].dtype == "category"

        if dataFile.expected_cols:
            for col in dataFile.expected_cols:
                assert col in data.columns


writedata = [
    DataFileParam(path="data/device_0.bin", expected_rows=1, expected_address=0, keep_type=True),
]


@mark.parametrize("dataFile", writedata)
def test_write(dataFile: DataFileParam):
    if dataFile.expected_address is None:
        raise AssertionError("expected address must be defined for all write tests")

    buffer = np.fromfile(dataFile.path, np.uint8)
    data = read(
        buffer,
        address=dataFile.expected_address,
        dtype=dataFile.expected_dtype,
        length=dataFile.expected_length,
        keep_type=dataFile.keep_type,
    )
    assert len(data) == dataFile.expected_rows
    write_buffer = format(data, address=dataFile.expected_address)
    assert np.array_equal(buffer, write_buffer)
