import mmap
import sys
from os import PathLike
from typing import Any, BinaryIO, Union

from numpy.typing import NDArray

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as _BufferLike
else:
    _BufferLike = Union[bytes, bytearray, memoryview, mmap.mmap, NDArray[Any]]

_FileLike = Union[str, PathLike[str], BinaryIO]
