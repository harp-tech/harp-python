import numpy as np
from pytest import mark
from harp.io import REFERENCE_EPOCH, MessageType
from harp.reader import create_reader
from tests.params import DeviceSchemaParam

testdata = [
    DeviceSchemaParam(
        path="data",
        expected_whoAmI=0,
        expected_registers=["DigitalInputMode"],
    ),
    DeviceSchemaParam(
        path="data/device.yml",
        expected_whoAmI=0,
        expected_registers=["DigitalInputMode"],
    ),
]


@mark.parametrize("schemaFile", testdata)
def test_create_reader(schemaFile: DeviceSchemaParam):
    reader = create_reader(schemaFile.path, epoch=REFERENCE_EPOCH)
    schemaFile.assert_schema(reader.device)

    whoAmI = reader.WhoAmI.read()
    assert reader.device.whoAmI == whoAmI.iloc[0, 0]
    assert whoAmI.index.dtype.type == np.datetime64

    whoAmI = reader.WhoAmI.read(epoch=None, keep_type=True)
    assert whoAmI.index.dtype.type == np.float64
    assert whoAmI.iloc[0, -1] == MessageType.READ.name
