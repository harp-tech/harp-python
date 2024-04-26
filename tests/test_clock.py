import numpy as np
from pytest import mark
from harp.clock import decode_harp_clock, align_timestamps_to_harp_clock
import warnings

# fmt: off
testinput = [
    {
        # contains two valid Harp clock barcodes
        'sample_numbers': np.array([
            0, 28832, 28892, 28922, 28952, 29012, 29072, 29132, 29192,
            29252, 29282, 29312, 29402, 29432, 29492, 29522, 29552, 29642,
            29702, 29732, 30002, 58835, 58865, 58925, 58955, 59015, 59075,
            59135, 59195, 59255, 59285, 59315, 59405, 59435, 59495, 59525,
            59555, 59645, 59705, 59735, 60005, 88838, 88928, 89018, 89078,
            89138, 89198, 89258, 89288, 89318, 89408, 89438, 89498, 89528,
            89558, 89648, 89708, 89738]),
        'states': np.array([
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        'sample_rate': 30000.,
        'expected_start_samples': np.array([28832, 58835]),
        'expected_harp_times': np.array([3806874, 3806875])
    },
    {
        # contains one valid and one invalid Harp clock barcode
        'sample_numbers': np.array([    
            0, 28833, 29103, 29133, 29283, 29343, 29373, 29433, 29463,
            29553, 29613, 29643, 29703, 29733, 30003, 58835, 58865, 58895,
            59105, 59135, 59285, 59345, 59375, 59435, 59465, 59555, 59615,
            59645, 59705, 59735, 59885, 59945, 59975, 60036, 60065, 60156,
            60216, 60246, 60306, 60336, 60606, 88958]),
        'states': np.array([
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        'sample_rate': 30000.,
        'expected_start_samples': np.array([28833, 58835]),
        'expected_harp_times': np.array([2600960, 2600961])
    },
    {
        # same as above, but with values in seconds
        'sample_numbers': np.array([
            0.        , 0.9611    , 0.9701    , 0.9711    , 0.9761    ,
            0.9781    , 0.9791    , 0.9811    , 0.9821    , 0.9851    ,
            0.9871    , 0.9881    , 0.9901    , 0.9911    , 1.0001    ,
            1.96116667, 1.96216667, 1.96316667, 1.97016667, 1.97116667,
            1.97616667, 1.97816667, 1.97916667, 1.98116667, 1.98216667,
            1.98516667, 1.98716667, 1.98816667, 1.99016667, 1.99116667,
            1.99616667, 1.99816667, 1.99916667, 2.0012    , 2.00216667,
            2.0052    , 2.0072    , 2.0082    , 2.0102    , 2.0112    ,
            2.0202    , 2.96526667]),
        'states': np.array([
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        'sample_rate' : 1,
        'expected_start_samples': np.array([0.9611, 1.96116667]),
        'expected_harp_times': np.array([2600960, 2600961])
    }
]
# fmt: on


@mark.parametrize("test_input", testinput)
def test_create_reader(test_input):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_samples, harp_times = decode_harp_clock(
            test_input["sample_numbers"],
            test_input["states"],
            sample_rate=test_input["sample_rate"],
            baud_rate=1000,
        )

    assert np.allclose(start_samples, test_input["expected_start_samples"])
    assert np.allclose(harp_times, test_input["expected_harp_times"])

    # test alignment for samples 1/2 second after anchors
    ts = start_samples + test_input["sample_rate"] * 0.5
    aligned_times = align_timestamps_to_harp_clock(ts, start_samples, harp_times)
    assert np.allclose(aligned_times, harp_times + 0.5)

    # test alignment for samples 1/2 second before anchors
    ts = start_samples - test_input["sample_rate"] * 0.5
    aligned_times = align_timestamps_to_harp_clock(ts, start_samples, harp_times)
    assert np.allclose(aligned_times, harp_times - 0.5)
