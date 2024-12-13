from harp.io import REFERENCE_EPOCH, MessageType, read
from harp.reader import create_reader
from harp.schema import read_schema
import warnings
import functools


__all__ = ["REFERENCE_EPOCH", "MessageType", "read", "create_reader", "read_schema"]


def deprecated(message):
    # This decorator is only available from the stdlib warnings module in Python 3.13
    # Making it available here for compatibility with older versions
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"Call to deprecated function {func.__name__}: {message}",
                category=DeprecationWarning,
                stacklevel=1,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
