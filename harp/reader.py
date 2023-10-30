import re
from functools import partial, reduce
from pandas import DataFrame
from typing import Iterable, Callable
from harp.model import Model, Register
from harp.io import read

_camel_to_snake_regex = re.compile(r"(?<!^)(?=[A-Z])")


class RegisterReader:
    register: Register
    read: Callable[[str], DataFrame]

    def __init__(self, register: Register, read: Callable[[str], DataFrame]) -> None:
        self.register = register
        self.read = read


class DeviceReader:
    model: Model
    registers: dict[str, RegisterReader]

    def __init__(self, model: Model, registers: dict[str, RegisterReader]) -> None:
        self.model = model
        self.registers = registers

    def __dir__(self) -> Iterable[str]:
        return self.registers.keys()

    def __getattr__(self, __name: str) -> RegisterReader:
        return self.registers[__name]


def compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def id_camel_to_snake(id: str):
    return _camel_to_snake_regex.sub("_", id).lower()


def keys_camel_to_snake(keys: Iterable[str]):
    return [id_camel_to_snake(k) for k in keys]


def create_bitreader(mask):
    return lambda xs: ((xs & mask) != 0)


def create_reader(device: Model):
    reg_readers = {
        name: create_register_reader(device, name) for name in device.registers.keys()
    }
    return DeviceReader(device, reg_readers)


def create_register_reader(device: Model, name: str):
    register = device.registers[name]
    reader = read

    if register.maskType is not None:
        key = register.maskType.root
        bitMask = device.bitMasks.get(key)
        if bitMask is not None:
            lookup = [
                (id_camel_to_snake(k), create_bitreader(v.root))
                for k, v in bitMask.bits.items()
            ]

            def unpack(df):
                return DataFrame({n: f(df[0]) for n, f in lookup}, index=df.index)

            reader = compose(unpack, reader)
            return RegisterReader(register, reader)

        groupMask = device.groupMasks.get(key)
        if groupMask is not None:
            name = id_camel_to_snake(name)
            lookup = {value.root: name for name, value in groupMask.values.items()}
            reader = partial(reader, columns=[name])
            reader = compose(lambda df: df.map(lambda x: lookup[x]), reader)
            return RegisterReader(register, reader)

    if register.payloadSpec is not None:
        columns = register.payloadSpec.keys()
        columns = keys_camel_to_snake(columns)
        reader = partial(reader, columns=columns)
        return RegisterReader(register, reader)

    columns = [id_camel_to_snake(name)]
    reader = partial(reader, columns=columns)
    return RegisterReader(register, reader)
