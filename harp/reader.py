import re
from math import log2
from os import PathLike
from functools import partial
from numpy import dtype
from pandas import DataFrame, Series
from typing import Any, BinaryIO, Iterable, Callable, Optional, Union
from pandas._typing import Axes
from harp.model import BitMask, GroupMask, Model, PayloadMember, Register
from harp.io import read
from harp.schema import read_schema

_camel_to_snake_regex = re.compile(r"(?<!^)(?=[A-Z])")


class RegisterReader:
    register: Register
    read: Callable[[Union[str, bytes, PathLike[Any], BinaryIO]], DataFrame]

    def __init__(
        self,
        register: Register,
        read: Callable[[Union[str, bytes, PathLike[Any], BinaryIO]], DataFrame],
    ) -> None:
        self.register = register
        self.read = read


class DeviceReader:
    device: Model
    registers: dict[str, RegisterReader]

    def __init__(self, device: Model, registers: dict[str, RegisterReader]) -> None:
        self.device = device
        self.registers = registers

    def __dir__(self) -> Iterable[str]:
        return self.registers.keys()

    def __getattr__(self, __name: str) -> RegisterReader:
        return self.registers[__name]


def _compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def _id_camel_to_snake(id: str):
    return _camel_to_snake_regex.sub("_", id).lower()


def _create_bit_parser(mask: int):
    def parser(xs: Series) -> Series:
        return (xs & mask) != 0

    return parser


def _create_bitmask_parser(bitMask: BitMask):
    lookup = [
        (_id_camel_to_snake(k), _create_bit_parser(int(v.root)))
        for k, v in bitMask.bits.items()
    ]

    def parser(df: DataFrame):
        return DataFrame({n: f(df[0]) for n, f in lookup}, index=df.index)

    return parser


def _create_groupmask_lookup(groupMask: GroupMask):
    return {int(v.root): n for n, v in groupMask.values.items()}


def _create_groupmask_parser(name: str, groupMask: GroupMask):
    lookup = _create_groupmask_lookup(groupMask)

    def parser(df: DataFrame):
        return DataFrame({name: df[0].map(lookup)})

    return parser


def _mask_shift(mask: int):
    lsb = mask & (~mask + 1)
    return int(log2(lsb))


def _create_payloadmember_parser(device: Model, member: PayloadMember):
    offset = member.offset
    if offset is None:
        offset = 0

    shift = 0
    if member.mask is not None:
        shift = _mask_shift(member.mask)

    lookup = None
    if member.maskType is not None:
        key = member.maskType.root
        groupMask = None if device.groupMasks is None else device.groupMasks.get(key)
        if groupMask is not None:
            lookup = _create_groupmask_lookup(groupMask)

    is_boolean = False
    if member.interfaceType is not None:
        is_boolean = member.interfaceType.root == "bool"

    def parser(df: DataFrame):
        series = df[offset]
        if member.mask is not None:
            series = series & member.mask
            if shift > 0:
                series = Series(series.values >> shift, series.index) # type: ignore
        if is_boolean:
            series = series != 0
        elif lookup is not None:
            series = series.map(lookup)
        return series

    return parser


def _create_register_reader(register: Register):
    def reader(
        file: Union[str, bytes, PathLike[Any], BinaryIO], columns: Optional[Axes] = None
    ):
        data = read(
            file,
            address=register.address,
            dtype=dtype(register.type),
            length=register.length,
            columns=columns,
        )
        return data

    return reader


def _create_register_parser(device: Model, name: str):
    register = device.registers[name]
    reader = _create_register_reader(register)

    if register.maskType is not None:
        key = register.maskType.root
        bitMask = None if device.bitMasks is None else device.bitMasks.get(key)
        if bitMask is not None:
            parser = _create_bitmask_parser(bitMask)
            reader = _compose(parser, reader)
            return RegisterReader(register, reader)

        groupMask = None if device.groupMasks is None else device.groupMasks.get(key)
        if groupMask is not None:
            parser = _create_groupmask_parser(name, groupMask)
            reader = _compose(parser, reader)
            return RegisterReader(register, reader)

    if register.payloadSpec is not None:
        payload_parsers = [
            (_id_camel_to_snake(key), _create_payloadmember_parser(device, member))
            for key, member in register.payloadSpec.items()
        ]

        def parser(df: DataFrame):
            return DataFrame({n: f(df) for n, f in payload_parsers}, index=df.index)

        reader = _compose(parser, reader)
        return RegisterReader(register, reader)

    columns = [_id_camel_to_snake(name)]
    reader = partial(reader, columns=columns)
    return RegisterReader(register, reader)


def create_reader(file: Union[str, PathLike], include_common_registers: bool = True):
    device = read_schema(file, include_common_registers)
    reg_readers = {
        name: _create_register_parser(device, name) for name in device.registers.keys()
    }
    return DeviceReader(device, reg_readers)
