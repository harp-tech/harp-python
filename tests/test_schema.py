from pytest import mark
from harp.schema import read_schema
from tests.params import DeviceSchemaParam

testdata = [
    DeviceSchemaParam(
        path="data/device.yml",
        expected_whoAmI=0,
        expected_registers=["DigitalInputMode"],
    )
]


@mark.parametrize("schemaFile", testdata)
def test_read_schema(schemaFile: DeviceSchemaParam):
    device = read_schema(schemaFile.path)
    schemaFile.assert_schema(device)
