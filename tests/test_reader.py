from pytest import mark
from harp.reader import create_reader
from tests.params import DeviceSchemaParam

testdata = [
    DeviceSchemaParam(
        path="data/device.yml",
        expected_whoAmI=0,
        expected_registers=["DigitalInputMode"],
    )
]


@mark.parametrize("schemaFile", testdata)
def test_create_reader(schemaFile: DeviceSchemaParam):
    reader = create_reader(schemaFile.path)
    schemaFile.assert_schema(reader.device)
