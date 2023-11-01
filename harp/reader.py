import re
from functools import partial
from pandas import DataFrame, Series
from typing import Iterable, Callable, Union
from harp.model import BitMask, GroupMask, MaskValueItem, Model, Register
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


def _keys_camel_to_snake(keys: Iterable[str]):
    return [_id_camel_to_snake(k) for k in keys]


def _create_bit_parser(mask: Union[int, MaskValueItem]):
    def parser(xs: Series) -> Series:
        return (xs & mask) != 0

    return parser


def _create_bitmask_parser(bitMask: BitMask):
    lookup = [
        (_id_camel_to_snake(k), _create_bit_parser(v.root))
        for k, v in bitMask.bits.items()
    ]

    def parser(df: DataFrame):
        return DataFrame({n: f(df[0]) for n, f in lookup}, index=df.index)

    return parser


def _create_groupmask_parser(name: str, groupMask: GroupMask):
    name = _id_camel_to_snake(name)
    lookup = {v.root: n for n, v in groupMask.values.items()}

    def parser(df: DataFrame):
        return DataFrame({name: df.map(lambda x: lookup[x])})

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
        columns = register.payloadSpec.keys()
        columns = _keys_camel_to_snake(columns)
        reader = partial(reader, columns=columns)
        return RegisterReader(register, reader)

    columns = [_id_camel_to_snake(name)]
    reader = partial(reader, columns=columns)
    return RegisterReader(register, reader)


def create_reader(device: Model):
    reg_readers = {
        name: _create_register_reader(device, name) for name in device.registers.keys()
    }
    return DeviceReader(device, reg_readers)
