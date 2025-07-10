from importlib import resources
from os import PathLike
from typing import TextIO, Union

import yaml

from harp.model import Model, Registers


def _convert_keys_to_strings(obj):
    """Recursively converts all dictionary keys to strings.
    This is necessary since pydantic deserialization from python objects
    seems to expect keys to always be strings."""
    if isinstance(obj, dict):
        return {str(k): _convert_keys_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_keys_to_strings(i) for i in obj]
    return obj


def _read_common_registers() -> Registers:
    if __package__ is None:
        raise ValueError("__package__ is None: unable to read common registers")

    file = resources.files(__package__) / "common.yml"
    with file.open("r") as fileIO:
        regs_raw = _convert_keys_to_strings(yaml.safe_load(fileIO.read()))
        return Registers.model_validate(regs_raw)


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
        schema_raw = _convert_keys_to_strings(yaml.safe_load(file.read()))
        schema = Model.model_validate(schema_raw)
        if "WhoAmI" not in schema.registers and include_common_registers:
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
