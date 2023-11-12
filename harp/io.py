from os import PathLike
from typing import Any, BinaryIO, Optional, Union
from pandas._typing import Axes
import numpy as np
import pandas as pd

_SECONDS_PER_TICK = 32e-6
payloadtypes = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.uint32),
    8: np.dtype(np.uint64),
    129: np.dtype(np.int8),
    130: np.dtype(np.int16),
    132: np.dtype(np.int32),
    136: np.dtype(np.int64),
    68: np.dtype(np.float32),
}


def read(
    file: Union[str, bytes, PathLike[Any], BinaryIO],
    address: Optional[int] = None,
    dtype: Optional[np.dtype] = None,
    length: Optional[int] = None,
    columns: Optional[Axes] = None,
):
    """
    Read single-register Harp data from the specified file.

    :param file: Open file object or filename containing binary data from
      a single device register.
    :param address: Expected register address. If specified, the address of
      the first message in the file is used for validation.
    :param dtype: Expected data type of the register payload. If specified, the
      payload type of the first message in the file is used for validation.
    :param length: Expected number of elements in register payload. If specified,
      the payload length of the first message in the file is used for validation.
    :param columns: The optional column labels to use for the data values.
    :return: A pandas data frame containing message data, sorted by time.
    """
    data = np.fromfile(file, dtype=np.uint8)
    if len(data) == 0:
        return pd.DataFrame(
            columns=columns, index=pd.Index([], dtype=np.float64, name="time")
        )

    if address is not None and address != data[2]:
        raise ValueError(f"expected address {address} but got {data[2]}")

    stride = data[1] + 2
    nrows = len(data) // stride
    payloadsize = stride - 12
    payloadtype = payloadtypes[data[4] & ~0x10]
    if dtype is not None and dtype != payloadtype:
        raise ValueError(f"expected payload type {dtype} but got {payloadtype}")

    elementsize = payloadtype.itemsize
    payloadshape = (nrows, payloadsize // elementsize)
    if length is not None and length != payloadshape[1]:
        raise ValueError(f"expected payload length {length} but got {payloadshape[1]}")

    seconds = np.ndarray(nrows, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    micros = np.ndarray(nrows, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = micros * _SECONDS_PER_TICK + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data,
        offset=11,
        strides=(stride, elementsize),
    )
    time = pd.Series(seconds)
    time.name = "time"
    return pd.DataFrame(payload, index=time, columns=columns)
