from datetime import datetime
from enum import IntEnum
from os import PathLike
from typing import Any, BinaryIO, Optional, Union
from pandas._typing import Axes
import numpy as np
import pandas as pd

"""The reference epoch for UTC harp time."""
REFERENCE_EPOCH = datetime(1904, 1, 1)


class MessageType(IntEnum):
    NA = 0
    READ = 1
    WRITE = 2
    EVENT = 3


_SECONDS_PER_TICK = 32e-6
_messagetypes = [type.name for type in MessageType]
_payloadtypes = {
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
    epoch: Optional[datetime] = None,
    keep_type: bool = False,
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
    :param epoch: Reference datetime at which time zero begins. If specified,
      the result data frame will have a datetime index.
    :param keep_type: Specifies whether to include a column with the message type.
    :return: A pandas data frame containing message data, sorted by time.
    """
    data = np.fromfile(file, dtype=np.uint8)
    if len(data) == 0:
        return pd.DataFrame(
            columns=columns, index=pd.Index([], dtype=np.float64, name="time")
        )

    if address is not None and address != data[2]:
        raise ValueError(f"expected address {address} but got {data[2]}")

    index = None
    stride = data[1] + 2
    nrows = len(data) // stride
    payloadtype = data[4]
    payloadoffset = 5
    if payloadtype & 0x10 != 0:
        seconds = np.ndarray(
            nrows, dtype=np.uint32, buffer=data, offset=payloadoffset, strides=stride
        )
        payloadoffset += 4
        micros = np.ndarray(
            nrows, dtype=np.uint16, buffer=data, offset=payloadoffset, strides=stride
        )
        payloadoffset += 2
        time = micros * _SECONDS_PER_TICK + seconds
        payloadtype = payloadtype & ~0x10
        if epoch is not None:
            time = epoch + pd.to_timedelta(time, "s") # type: ignore
        index = pd.Series(time)
        index.name = "time"

    payloadsize = stride - payloadoffset - 1
    payloadtype = _payloadtypes[payloadtype]
    if dtype is not None and dtype != payloadtype:
        raise ValueError(f"expected payload type {dtype} but got {payloadtype}")

    elementsize = payloadtype.itemsize
    payloadshape = (nrows, payloadsize // elementsize)
    if length is not None and length != payloadshape[1]:
        raise ValueError(f"expected payload length {length} but got {payloadshape[1]}")

    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data,
        offset=payloadoffset,
        strides=(stride, elementsize),
    )

    result = pd.DataFrame(payload, index=index, columns=columns)
    if keep_type:
        msgtype = np.ndarray(
            nrows, dtype=np.uint8, buffer=data, offset=0, strides=stride
        )
        msgtype = pd.Categorical.from_codes(msgtype, categories=_messagetypes) # type: ignore
        result["type"] = msgtype
    return result
