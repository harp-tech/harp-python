import numpy as np
import pandas as pd

_SECONDS_PER_TICK = 32e6
payloadtypes = {
    1 : np.dtype(np.uint8),
    2 : np.dtype(np.uint16),
    4 : np.dtype(np.uint32),
    8 : np.dtype(np.uint64),
    129 : np.dtype(np.int8),
    130 : np.dtype(np.int16),
    132 : np.dtype(np.int32),
    136 : np.dtype(np.int64),
    68 : np.dtype(np.float32)
}

def read(file: str, columns=None):
    '''
    Read single-register Harp data from the specified file.
    
    :param str file: The path to a Harp binary file containing data from a single device register.
    :param str or array-like names: The optional column labels to use for the data values.
    :return: A pandas data frame containing harp event data, sorted by time.
    '''
    data = np.fromfile(file, dtype=np.uint8)
    if len(data) == 0:
        return pd.DataFrame(
            columns=columns,
            index=pd.Index([], dtype=np.float64, name='time'))

    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    micros = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = micros * _SECONDS_PER_TICK + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data, offset=11,
        strides=(stride, elementsize))
    time = pd.Series(seconds)
    time.name = 'time'
    return pd.DataFrame(payload, index=time, columns=columns)