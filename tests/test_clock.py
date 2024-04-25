import numpy as np
from pytest import mark
from harp.clock import decode_harp_clock

# fmt: off
testinput = [
    {
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
        'expected_start_samples': np.array([28832, 58835]),
        'expected_harp_times': np.array([3806874, 3806875])
    },
    {
        'sample_numbers': np.array([    
            0, 28833, 29103, 29133, 29283, 29343, 29373, 29433, 29463,
            29553, 29613, 29643, 29703, 29733, 30003, 58835, 58865, 58895,
            59105, 59135, 59285, 59345, 59375, 59435, 59465, 59555, 59615,
            59645, 59705, 59735, 59885, 59945, 59975, 60036, 60065, 60156,
            60216, 60246, 60306, 60336, 60606, 88958]),
        'states': np.array([
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        'expected_start_samples': np.array([28833, 58835]),
        'expected_harp_times': np.array([2600960, 2600961])
    }
]
# fmt: on


@mark.parametrize("test_input", testinput)
def test_create_reader(test_input):
    start_samples, harp_times = decode_harp_clock(
        test_input["sample_numbers"], test_input["states"]
    )

    assert np.allclose(start_samples, test_input["expected_start_samples"])
    assert np.allclose(harp_times, test_input["expected_harp_times"])
