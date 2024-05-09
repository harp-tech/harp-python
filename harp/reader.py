import os
from collections import UserDict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from math import log2
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Optional, Protocol, Union

from numpy import dtype
from pandas import DataFrame, Series
from pandas._typing import Axes

from harp.io import MessageType, read
from harp.model import BitMask, GroupMask, Model, PayloadMember, Register
from harp.schema import read_schema


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
    ) -> DataFrame: ...


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


class RegisterMap(UserDict[str, RegisterReader]):
    _address_map: Mapping[int, RegisterReader]

    def __init__(self, registers: Mapping[str, RegisterReader]) -> None:
        super().__init__(registers)
        self._address_map = {value.register.address: value for value in registers.values()}

    def __getitem__(self, key: Union[str, int]) -> RegisterReader:
        if isinstance(key, int):
            return self._address_map[key]
        else:
            return super().__getitem__(key)


class DeviceReader:
    device: Model
    registers: RegisterMap

    def __init__(self, device: Model, registers: Mapping[str, RegisterReader]) -> None:
        self.device = device
        self.registers = RegisterMap(registers)

    def __dir__(self) -> Iterable[str]:
        return self.registers.keys()

    def __getattr__(self, __name: str) -> RegisterReader:
        return self.registers[__name]

    @staticmethod
    def from_file(
        filepath: PathLike,
        base_path: Optional[PathLike] = None,
        include_common_registers: bool = True,
        epoch: Optional[datetime] = None,
        keep_type: bool = False,
    ) -> "DeviceReader":
        """Creates a device reader object from the specified schema yml file.

        Parameters
        ----------
        filepath
            A path to the device yml schema describing the device.
        base_path
            The path to attempt to resolve the location of data files.
        include_common_registers
            Specifies whether to include the set of Harp common registers in the
            parsed device schema object. If a parsed device schema object is provided,
            this parameter is ignored.
        epoch
            The default reference datetime at which time zero begins. If specified,
            the data frames returned by each register reader will have a datetime index.
        keep_type
            Specifies whether to include a column with the message type by default.

        Returns
        -------
            A device reader object which can be used to read binary data for each
            register or to access metadata about each register. Individual registers
            can be accessed using dot notation using the name of the register as the
            key.
        """

        device = read_schema(filepath, include_common_registers)
        if base_path is None:
            path = Path(filepath).absolute().resolve()
            base_path = path.parent / device.device
        else:
            base_path = Path(base_path).absolute().resolve() / device.device

        reg_readers = {
            name: _create_register_parser(
                device, name, _ReaderParams(base_path, epoch, keep_type)
            )
            for name in device.registers.keys()
        }
        return DeviceReader(device, reg_readers)

    @staticmethod
    def from_url(
        url: str,
        base_path: Optional[PathLike] = None,
        include_common_registers: bool = True,
        epoch: Optional[datetime] = None,
        keep_type: bool = False,
        timeout: int = 5,
    ) -> "DeviceReader":
        """Creates a device reader object from a url pointing to a device.yml file.

        Parameters
        ----------
        url
            The url pointing to the device.yml schema describing the device.
        base_path
            The path to attempt to resolve the location of data files.
        include_common_registers
            Specifies whether to include the set of Harp common registers in the
            parsed device schema object. If a parsed device schema object is provided,
            this parameter is ignored.
        epoch
            The default reference datetime at which time zero begins. If specified,
            the data frames returned by each register reader will have a datetime index.
        keep_type
            Specifies whether to include a column with the message type by default.
        timeout
            The number of seconds to wait for the server to send data before giving up.
        Returns
        -------
            A device reader object which can be used to read binary data for each
            register or to access metadata about each register. Individual registers
            can be accessed using dot notation using the name of the register as the
            key.
        """

        response = requests.get(url, timeout=timeout)
        text = response.text

        device = read_schema(text, include_common_registers)
        if base_path is None:
            base_path = Path(device.device).absolute().resolve()
        else:
            base_path = Path(base_path).absolute().resolve()

        reg_readers = {
            name: _create_register_parser(
                device, name, _ReaderParams(base_path, epoch, keep_type)
            )
            for name in device.registers.keys()
        }
        return DeviceReader(device, reg_readers)

    @staticmethod
    def from_str(
        schema: str,
        base_path: Optional[PathLike] = None,
        include_common_registers: bool = True,
        epoch: Optional[datetime] = None,
        keep_type: bool = False,
    ) -> "DeviceReader":
        """Creates a device reader object from a string containing a device.yml schema.

        Parameters
        ----------
        schema
            The string containing the device.yml schema describing the device.
        base_path
            The path to attempt to resolve the location of data files.
        include_common_registers
            Specifies whether to include the set of Harp common registers in the
            parsed device schema object. If a parsed device schema object is provided,
            this parameter is ignored.
        epoch
            The default reference datetime at which time zero begins. If specified,
            the data frames returned by each register reader will have a datetime index.
        keep_type
            Specifies whether to include a column with the message type by default.

        Returns
        -------
            A device reader object which can be used to read binary data for each
            register or to access metadata about each register. Individual registers
            can be accessed using dot notation using the name of the register as the
            key.
        """

        device = read_schema(schema, include_common_registers)
        if base_path is None:
            base_path = Path(device.device).absolute().resolve()
        else:
            base_path = Path(base_path).absolute().resolve()

        reg_readers = {
            name: _create_register_parser(
                device, name, _ReaderParams(base_path, epoch, keep_type)
            )
            for name in device.registers.keys()
        }
        return DeviceReader(device, reg_readers)

    @staticmethod
    def from_model(
        model: Model,
        base_path: Optional[PathLike] = None,
        epoch: Optional[datetime] = None,
        keep_type: bool = False,
    ) -> "DeviceReader":
        """Creates a device reader object from a parsed device schema object.

        Parameters
        ----------
        model
            The parsed device schema object describing the device.
        base_path
            The path to attempt to resolve the location of data files.
        epoch
            The default reference datetime at which time zero begins. If specified,
            the data frames returned by each register reader will have a datetime index.
        keep_type
            Specifies whether to include a column with the message type by default.

        Returns
        -------
            A device reader object which can be used to read binary data for each
            register or to access metadata about each register. Individual registers
            can be accessed using dot notation using the name of the register as the
            key.
        """

        if base_path is None:
            base_path = Path(model.device).absolute().resolve()
        else:
            base_path = Path(base_path).absolute().resolve()

        reg_readers = {
            name: _create_register_parser(
                model, name, _ReaderParams(base_path, epoch, keep_type)
            )
            for name in model.registers.keys()
        }
        return DeviceReader(model, reg_readers)

    @staticmethod
    def from_dataset(
        dataset: PathLike,
        include_common_registers: bool = True,
        epoch: Optional[datetime] = None,
        keep_type: bool = False,
    ) -> "DeviceReader":
        """Creates a device reader object from the specified dataset folder.

        Parameters
        ----------
        dataset
            A path to the dataset folder containing a device.yml schema describing the device.
        include_common_registers
            Specifies whether to include the set of Harp common registers in the
            parsed device schema object. If a parsed device schema object is provided,
            this parameter is ignored.
        epoch
            The default reference datetime at which time zero begins. If specified,
            the data frames returned by each register reader will have a datetime index.
        keep_type
            Specifies whether to include a column with the message type by default.

        Returns
        -------
            A device reader object which can be used to read binary data for each
            register or to access metadata about each register. Individual registers
            can be accessed using dot notation using the name of the register as the
            key.
        """

        path = Path(dataset).absolute().resolve()
        is_dir = os.path.isdir(path)
        if is_dir:
            filepath = path / "device.yml"
            return DeviceReader.from_file(
                filepath=filepath,
                base_path=path,
                include_common_registers=include_common_registers,
                epoch=epoch,
                keep_type=keep_type)
        else:
            raise ValueError("The dataset must be a directory containing a device.yml file.")


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
        type_col = df.get(MessageType.__name__)
        if type_col is not None:
            result[MessageType.__name__] = type_col
        return result

    return parser


def _create_bit_parser(mask: int):
    def parser(xs: Series) -> Series:
        return (xs & mask) != 0

    return parser


def _create_bitmask_parser(bitMask: BitMask):
    lookup = [(k, _create_bit_parser(int(v.root))) for k, v in bitMask.bits.items()]

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
            (key, _create_payloadmember_parser(device, member))
            for key, member in register.payloadSpec.items()
        ]

        def parser(df: DataFrame):
            return DataFrame({n: f(df) for n, f in payload_parsers}, index=df.index)

        reader = _compose_parser(parser, reader, params)
        return RegisterReader(register, reader)

    reader = partial(reader, columns=[name])
    return RegisterReader(register, reader)

@deprecated("This function is deprecated. Use DeviceReader.from_file, DeviceReader.from_url, DeviceReader.from_str, and DeviceReader.from_model instead.")
def create_reader(
    device: Union[str, PathLike, Model],
    include_common_registers: bool = True,
    epoch: Optional[datetime] = None,
    keep_type: bool = False,
):
    """Creates a device reader object from the specified dataset or schema.

    Parameters
    ----------
    device
        A path to the device schema, dataset folder, or parsed device schema object
        describing the device.
    include_common_registers
        Specifies whether to include the set of Harp common registers in the
        parsed device schema object. If a parsed device schema object is provided,
        this parameter is ignored.
    epoch
        The default reference datetime at which time zero begins. If specified,
        the data frames returned by each register reader will have a datetime index.
    keep_type
        Specifies whether to include a column with the message type by default.

    Returns
    -------
        A device reader object which can be used to read binary data for each
        register or to access metadata about each register. Individual registers
        can be accessed using dot notation using the name of the register as the
        key.
    """
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
        name: _create_register_parser(device, name, _ReaderParams(base_path, epoch, keep_type))
        for name in device.registers.keys()
    }
    return DeviceReader(device, reg_readers)
