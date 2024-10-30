import mmap
import sys
from typing import Any, Union

from numpy.typing import NDArray

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as BufferLike
else:
    BufferLike = Union[bytes, bytearray, memoryview, mmap.mmap, NDArray[Any]]
