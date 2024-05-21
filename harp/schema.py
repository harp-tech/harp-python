from os import PathLike
from typing import TextIO, Union
from harp.model import Model, Registers
from pydantic_yaml import parse_yaml_raw_as
from importlib import resources


def _read_common_registers() -> Registers:
    file = resources.files(__package__) / "common.yml"
    with file.open("rt") as fileIO:
        return parse_yaml_raw_as(Registers, fileIO.read())


def read_schema(file: Union[str, PathLike, TextIO], include_common_registers: bool = True) -> Model:
    """Read and parse a device schema from the specified file.

    Parameters
    ----------
    file
        Open file object or filename containing a YAML text stream describing
        a device schema.
    include_common_registers
        Specifies whether to include the set of Harp common registers in the
        returned device schema object.

    Returns
    -------
        A Pydantic model object representing the Harp device schema.
    """

    if isinstance(file, (str, PathLike)):
        with open(file) as fileIO:
            return read_schema(fileIO)
    else:
        schema = parse_yaml_raw_as(Model, file.read())
        if not "WhoAmI" in schema.registers and include_common_registers:
            common = _read_common_registers()
            schema.registers = dict(common.registers, **schema.registers)
            if common.bitMasks:
                schema.bitMasks = (
                    common.bitMasks if schema.bitMasks is None else dict(common.bitMasks, **schema.bitMasks)
                )
            if common.groupMasks:
                schema.groupMasks = (
                    common.groupMasks
                    if schema.groupMasks is None
                    else dict(common.groupMasks, **schema.groupMasks)
                )
        return schema
