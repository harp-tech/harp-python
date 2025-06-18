from unittest.mock import Mock, patch

import pandas as pd
from pytest import mark

from harp.io import REFERENCE_EPOCH, MessageType
from harp.reader import DeviceReader, Model, create_reader, read_schema
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


def helper_test_reader(reader: DeviceReader, schemaFile: DeviceSchemaParam) -> None:
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


@mark.parametrize("schemaFile", testdata)
@mark.filterwarnings("ignore:Call to deprecated")
def test_create_reader(schemaFile: DeviceSchemaParam):
    reader = create_reader(schemaFile.path, epoch=REFERENCE_EPOCH)
    helper_test_reader(reader, schemaFile)


@mark.parametrize("schemaFile", testdata)
def test_create_reader_from_file(schemaFile: DeviceSchemaParam):
    reader = DeviceReader.from_file("./tests/data/device.yml", epoch=REFERENCE_EPOCH)
    helper_test_reader(reader, schemaFile)


@mark.parametrize("schemaFile", testdata)
def test_create_reader_from_dataset(schemaFile: DeviceSchemaParam):
    reader = DeviceReader.from_dataset("./tests/data", epoch=REFERENCE_EPOCH)
    helper_test_reader(reader, schemaFile)


@mark.parametrize("schemaFile", testdata)
def test_create_reader_from_str(schemaFile: DeviceSchemaParam):
    with open("./tests/data/device.yml", "r", encoding="utf-8") as f:
        reader = DeviceReader.from_str(f.read(), base_path="./tests/data/", epoch=REFERENCE_EPOCH)
        helper_test_reader(reader, schemaFile)


@mark.parametrize("schemaFile", testdata)
def test_create_reader_from_model(schemaFile: DeviceSchemaParam):
    model = read_schema("./tests/data/device.yml", include_common_registers=True)
    reader = DeviceReader.from_model(model=model, base_path="./tests/data/", epoch=REFERENCE_EPOCH)
    helper_test_reader(reader, schemaFile)


@mark.parametrize("schemaFile", testdata)
@patch("requests.get")
def test_create_reader_from_url(mock_url_get, schemaFile: DeviceSchemaParam):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = open("./tests/data/device.yml", "r", encoding="utf-8").read()

    mock_url_get.return_value = mock_response

    reader = DeviceReader.from_url("mocked_url", base_path="./tests/data/", epoch=REFERENCE_EPOCH)
    helper_test_reader(reader, schemaFile)
