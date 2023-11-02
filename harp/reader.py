import re
from math import log2
from functools import partial
from pandas import DataFrame, Series
from typing import Iterable, Callable, Union
from harp.model import BitMask, GroupMask, MaskValueItem, Model, PayloadMember, Register
from harp.io import read

_camel_to_snake_regex = re.compile(r"(?<!^)(?=[A-Z])")


class RegisterReader:
    register: Register
    read: Callable[[str], DataFrame]

    def __init__(self, register: Register, read: Callable[[str], DataFrame]) -> None:
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

    shift = None
    if member.mask is not None:
        shift = _mask_shift(member.mask)

    lookup = None
    if member.maskType is not None:
        key = member.maskType.root
        groupMask = device.groupMasks.get(key)
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
                series = Series(series.values >> shift, series.index)
        if is_boolean:
            series = series != 0
        elif lookup is not None:
            series = series.map(lookup)
        return series

    return parser


def _create_register_reader(device: Model, name: str):
    register = device.registers[name]
    reader = read

    if register.maskType is not None:
        key = register.maskType.root
        bitMask = device.bitMasks.get(key)
        if bitMask is not None:
            parser = _create_bitmask_parser(bitMask)
            reader = _compose(parser, reader)
            return RegisterReader(register, reader)

        groupMask = device.groupMasks.get(key)
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


def create_reader(device: Model):
    reg_readers = {
        name: _create_register_reader(device, name) for name in device.registers.keys()
    }
    return DeviceReader(device, reg_readers)
