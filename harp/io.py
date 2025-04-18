from datetime import datetime
from enum import IntEnum
from os import PathLike
from typing import Any, BinaryIO, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas._typing import Axes  # pyright: ignore[reportPrivateImportUsage]

from harp.typing import _BufferLike, _FileLike

REFERENCE_EPOCH = datetime(1904, 1, 1)
"""The reference epoch for UTC harp time."""


class MessageType(IntEnum):
    """Specifies the type of a Harp message."""

    NA = 0
    READ = 1
    WRITE = 2
    EVENT = 3


_SECONDS_PER_TICK = 32e-6
_PAYLOAD_TIMESTAMP_MASK = 0x10
_messagetypes = [type.name for type in MessageType]
_dtypefrompayloadtype = {
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
_payloadtypefromdtype = {v: k for k, v in _dtypefrompayloadtype.items()}


def read(
    file_or_buf: Union[_FileLike, _BufferLike],
    address: Optional[int] = None,
    dtype: Optional[np.dtype] = None,
    length: Optional[int] = None,
    columns: Optional[Axes] = None,
    epoch: Optional[datetime] = None,
    keep_type: bool = False,
):
    """Read single-register Harp data from the specified file or buffer.

    Parameters
    ----------
    file_or_buf
        File path, open file object, or buffer containing binary data from
        a single device register.
    address
        Expected register address. If specified, the address of
        the first message is used for validation.
    dtype
        Expected data type of the register payload. If specified, the
        payload type of the first message is used for validation.
    length
        Expected number of elements in register payload. If specified, the
        payload length of the first message is used for validation.
    columns
        The optional column labels to use for the data values.
    epoch
        Reference datetime at which time zero begins. If specified,
        the result data frame will have a datetime index.
    keep_type
        Specifies whether to include a column with the message type.

    Returns
    -------
        A pandas data frame containing message data, sorted by time.
    """
    if isinstance(file_or_buf, (str, PathLike, BinaryIO)) or hasattr(file_or_buf, "readinto"):
        # TODO: in the below we ignore the type as otherwise
        # we have no way to runtime check _IOProtocol
        data = np.fromfile(file_or_buf, dtype=np.uint8)  # type: ignore
    else:
        data = np.frombuffer(file_or_buf, dtype=np.uint8)

    if len(data) == 0:
        return pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex([], name="Time")
            if epoch
            else pd.Index([], dtype=np.float64, name="Time"),
        )

    if address is not None and address != data[2]:
        raise ValueError(f"expected address {address} but got {data[2]}")

    index = None
    stride = int(data[1] + 2)
    nrows = len(data) // stride
    payloadtype = data[4]
    payloadoffset = 5
    if payloadtype & _PAYLOAD_TIMESTAMP_MASK != 0:
        seconds = np.ndarray(nrows, dtype=np.uint32, buffer=data, offset=payloadoffset, strides=stride)
        payloadoffset += 4
        micros = np.ndarray(nrows, dtype=np.uint16, buffer=data, offset=payloadoffset, strides=stride)
        payloadoffset += 2
        time = micros * _SECONDS_PER_TICK + seconds
        payloadtype = payloadtype & ~np.uint8(_PAYLOAD_TIMESTAMP_MASK)
        if epoch is not None:
            time = epoch + pd.to_timedelta(time, "s")  # type: ignore
        index = pd.Series(time)
        index.name = "Time"

    payloadsize = stride - payloadoffset - 1
    payloadtype = _dtypefrompayloadtype[payloadtype]
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
        msgtype = np.ndarray(nrows, dtype=np.uint8, buffer=data, offset=0, strides=stride)
        msgtype = pd.Categorical.from_codes(msgtype, categories=_messagetypes)  # type: ignore
        result[MessageType.__name__] = msgtype
    return result


def to_file(
    data: pd.DataFrame,
    file: _FileLike,
    address: int,
    dtype: Optional[np.dtype] = None,
    length: Optional[int] = None,
    port: Optional[int] = None,
    epoch: Optional[datetime] = None,
    message_type: Optional[MessageType] = None,
):
    """Write single-register Harp data to the specified file.

    Parameters
    ----------
    data
        Pandas data frame containing message payload.
    file
        File path, or open file object in which to store binary data from
        a single device register.
    address
        Register address used to identify all formatted Harp messages.
    dtype
        Data type of the register payload. If specified, all data will
        be converted before formatting the binary payload.
    length
        Expected number of elements in register payload. If specified, the
        number of columns in the input data frame is validated.
    port
        Optional port value used for all formatted Harp messages.
    epoch
        Reference datetime at which time zero begins. If specified,
        the input data frame must have a datetime index.
    message_type
        Optional message type used for all formatted Harp messages.
        If not specified, data must contain a MessageType column.
    """
    buffer = to_buffer(data, address, dtype, port, length, epoch, message_type)
    buffer.tofile(file)


def to_buffer(
    data: pd.DataFrame,
    address: int,
    dtype: Optional[np.dtype] = None,
    length: Optional[int] = None,
    port: Optional[int] = None,
    epoch: Optional[datetime] = None,
    message_type: Optional[MessageType] = None,
) -> npt.NDArray[np.uint8]:
    """Convert single-register Harp data to a flat binary buffer.

    Parameters
    ----------
    data
        Pandas data frame containing message payload.
    address
        Register address used to identify all formatted Harp messages.
    dtype
        Data type of the register payload. If specified, all data will
        be converted before formatting the binary payload.
    length
        Expected number of elements in register payload. If specified, the
        number of columns in the input data frame is validated.
    port
        Optional port value used for all formatted Harp messages.
    epoch
        Reference datetime at which time zero begins. If specified,
        the input data frame must have a datetime index.
    message_type
        Optional message type used for all formatted Harp messages.
        If not specified, data must contain a MessageType column.

    Returns
    -------
        An array object containing message data formatted according
        to the Harp binary protocol.
    """
    nrows = len(data)
    if nrows == 0:
        return np.empty(0, dtype=np.uint8)

    if MessageType.__name__ in data.columns:
        msgtype = data[MessageType.__name__].cat.codes
        payload = data[data.columns.drop(MessageType.__name__)].values
    elif message_type is not None:
        msgtype = message_type
        payload = data.values
    else:
        raise ValueError(f"message type must be specified either in the data or as argument")

    time = data.index
    is_timestamped = True
    if epoch is not None:
        if not isinstance(time, pd.DatetimeIndex):
            raise ValueError(f"expected datetime index to encode with epoch but got {time.inferred_type}")
        time = (time - epoch).total_seconds()
    elif isinstance(time, pd.RangeIndex):
        is_timestamped = False

    if dtype is not None:
        payload = payload.astype(dtype, copy=False)

    ncols = payload.shape[1]
    if length is not None and ncols != length:
        raise ValueError(f"expected payload length {length} but got {ncols}")

    if port is None:
        port = 255

    payloadtype = _payloadtypefromdtype[payload.dtype]
    payloadlength = ncols * payload.dtype.itemsize
    stride = payloadlength + 6
    if is_timestamped:
        payloadtype |= _PAYLOAD_TIMESTAMP_MASK
        stride += 6

    buffer = np.empty((nrows, stride), dtype=np.uint8)
    buffer[:, 0] = msgtype
    buffer[:, 1:5] = [stride - 2, address, port, payloadtype]

    payloadoffset = 5
    if is_timestamped:
        seconds = time.astype(np.uint32)
        micros = np.around(((time - seconds) / _SECONDS_PER_TICK).values).astype(np.uint16)
        buffer[:, 5:9] = np.ndarray((nrows, 4), dtype=np.uint8, buffer=seconds.values)
        buffer[:, 9:11] = np.ndarray((nrows, 2), dtype=np.uint8, buffer=micros)
        payloadoffset += 6

    payloadstop = payloadoffset + payloadlength
    buffer[:, payloadoffset:payloadstop] = np.ndarray(
        (nrows, payloadlength), dtype=np.uint8, buffer=np.ascontiguousarray(payload)
    )
    buffer[:, -1] = np.sum(buffer[:, 0:-1], axis=1, dtype=np.uint8)
    return buffer.reshape(-1)
