import pandas as pd
from pytest import mark

from harp.io import REFERENCE_EPOCH, MessageType
from harp.reader import create_reader
from tests.params import DeviceSchemaParam

testdata = [
    DeviceSchemaParam(
        path="data",
        expected_whoAmI=0,
        expected_registers=["AnalogData"],
    ),
    DeviceSchemaParam(
        path="data/device.yml",
        expected_whoAmI=0,
        expected_registers=["AnalogData"],
    ),
    DeviceSchemaParam(
        path="data/device.yml",
        expected_whoAmI=0,
        expected_registers=["AnalogDataPayloadSpec"],
    ),
]


@mark.parametrize("schemaFile", testdata)
def test_create_reader(schemaFile: DeviceSchemaParam):
    reader = create_reader(schemaFile.path, epoch=REFERENCE_EPOCH)
    schemaFile.assert_schema(reader.device)

    whoAmI = reader.WhoAmI.read()
    assert reader.device.whoAmI == whoAmI.iloc[0, 0]
    assert isinstance(whoAmI.index, pd.DatetimeIndex)

    whoAmI = reader.WhoAmI.read(epoch=None, keep_type=True)
    assert isinstance(whoAmI.index, pd.Index)
    assert whoAmI.iloc[0, -1] == MessageType.READ.name

    if schemaFile.expected_registers:
        for register_name in schemaFile.expected_registers:
            data = reader.registers[register_name].read()
            assert isinstance(data.index, pd.DatetimeIndex)
