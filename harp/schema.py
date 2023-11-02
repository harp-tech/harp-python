from os import PathLike
from typing import TextIO, Union
from harp.model import Model
from pydantic_yaml import parse_yaml_raw_as

def read_schema(file: Union[str, PathLike, TextIO]) -> Model:
    try:
        with open(file) as fileIO:
            return read_schema(fileIO)
    except TypeError:
        return parse_yaml_raw_as(Model, file.read())
