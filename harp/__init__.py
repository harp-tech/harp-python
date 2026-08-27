import warnings

warnings.warn(
    "The 'harp-python' package is deprecated and will no longer receive updates. "
    "Please migrate to the new 'harp-data' package: pip install harp-data",
    DeprecationWarning,
    stacklevel=2,
)

from harp.io import REFERENCE_EPOCH, MessageType, read, to_buffer, to_file
from harp.reader import create_reader
from harp.schema import read_schema

__all__ = ["REFERENCE_EPOCH", "MessageType", "read", "to_buffer", "to_file", "create_reader", "read_schema"]
