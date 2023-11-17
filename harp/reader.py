import os
import re
from math import log2
from os import PathLike
from pathlib import Path
from datetime import datetime
from functools import partial
from dataclasses import dataclass
from numpy import dtype
from pandas import DataFrame, Series
from typing import Any, BinaryIO, Callable, Iterable, Optional, Protocol, Union
from pandas._typing import Axes
from harp.model import BitMask, GroupMask, Model, PayloadMember, Register
from harp.io import read
from harp.schema import read_schema

_camel_to_snake_regex = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass
class _ReaderParams:
    path: Path
    epoch: Optional[datetime] = None
    keep_type: bool = False


class _ReadRegister(Protocol):
    def __call__(
        self,
        file: Optional[Union[str, bytes, PathLike[Any], BinaryIO]] = None,
        epoch: Optional[datetime] = None,
        keep_type: bool = False,
    ) -> DataFrame:
        ...


class RegisterReader:
    register: Register
    read: _ReadRegister

    def __init__(
        self,
        register: Register,
        read: _ReadRegister,
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


def _compose_parser(
    f: Callable[[DataFrame], DataFrame],
    g: Callable[..., DataFrame],
    params: _ReaderParams,
) -> Callable[..., DataFrame]:
    def parser(
        file: Optional[Union[str, bytes, PathLike[Any], BinaryIO]] = None,
        columns: Optional[Axes] = None,
        epoch: Optional[datetime] = params.epoch,
        keep_type: bool = params.keep_type,
    ):
        df = g(file, columns, epoch, keep_type)
        result = f(df)
        type_col = df.get("type")
        if type_col is not None:
            result["type"] = type_col
        return result

    return parser


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
                series = Series(series.values >> shift, series.index)  # type: ignore
        if is_boolean:
            series = series != 0
        elif lookup is not None:
            series = series.map(lookup)
        return series

    return parser


def _create_register_reader(register: Register, params: _ReaderParams):
    def reader(
        file: Optional[Union[str, bytes, PathLike[Any], BinaryIO]] = None,
        columns: Optional[Axes] = None,
        epoch: Optional[datetime] = params.epoch,
        keep_type: bool = params.keep_type,
    ):
        if file is None:
            file = f"{params.path}_{register.address}.bin"

        data = read(
            file,
            address=register.address,
            dtype=dtype(register.type),
            length=register.length,
            columns=columns,
            epoch=epoch,
            keep_type=keep_type,
        )
        return data

    return reader


def _create_register_parser(device: Model, name: str, params: _ReaderParams):
    register = device.registers[name]
    reader = _create_register_reader(register, params)

    if register.maskType is not None:
        key = register.maskType.root
        bitMask = None if device.bitMasks is None else device.bitMasks.get(key)
        if bitMask is not None:
            parser = _create_bitmask_parser(bitMask)
            reader = _compose_parser(parser, reader, params)
            return RegisterReader(register, reader)

        groupMask = None if device.groupMasks is None else device.groupMasks.get(key)
        if groupMask is not None:
            parser = _create_groupmask_parser(name, groupMask)
            reader = _compose_parser(parser, reader, params)
            return RegisterReader(register, reader)

    if register.payloadSpec is not None:
        payload_parsers = [
            (_id_camel_to_snake(key), _create_payloadmember_parser(device, member))
            for key, member in register.payloadSpec.items()
        ]

        def parser(df: DataFrame):
            return DataFrame({n: f(df) for n, f in payload_parsers}, index=df.index)

        reader = _compose_parser(parser, reader, params)
        return RegisterReader(register, reader)

    columns = [_id_camel_to_snake(name)]
    reader = partial(reader, columns=columns)
    return RegisterReader(register, reader)


def create_reader(
    device: Union[str, PathLike, Model],
    include_common_registers: bool = True,
    epoch: Optional[datetime] = None,
    keep_type: bool = False,
):
    if isinstance(device, Model):
        base_path = Path(device.device)
    else:
        path = Path(device).absolute().resolve()
        is_dir = os.path.isdir(path)
        if is_dir:
            device = path / "device.yml"

        device = read_schema(device, include_common_registers)
        base_path = path / device.device if is_dir else path.parent / device.device

    reg_readers = {
        name: _create_register_parser(
            device, name, _ReaderParams(base_path, epoch, keep_type)
        )
        for name in device.registers.keys()
    }
    return DeviceReader(device, reg_readers)
