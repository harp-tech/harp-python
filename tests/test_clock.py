import numpy as np
from pytest import mark
from harp.clock import decode_harp_clock, align_timestamps_to_anchor_points
import warnings

# fmt: off
testinput = [
    {
        # contains two valid Harp clock barcodes
        'timestamps': np.array([
            0.        , 0.96106667, 0.96306667, 0.96406667, 0.96506667,
            0.96706667, 0.96906667, 0.97106667, 0.97306667, 0.97506667,
            0.97606667, 0.97706667, 0.98006667, 0.98106667, 0.98306667,
            0.98406667, 0.98506667, 0.98806667, 0.99006667, 0.99106667,
            1.00006667, 1.96116667, 1.96216667, 1.96416667, 1.96516667,
            1.96716667, 1.96916667, 1.97116667, 1.97316667, 1.97516667,
            1.97616667, 1.97716667, 1.98016667, 1.98116667, 1.98316667,
            1.98416667, 1.98516667, 1.98816667, 1.99016667, 1.99116667,
            2.00016667, 2.96126667, 2.96426667, 2.96726667, 2.96926667,
            2.97126667, 2.97326667, 2.97526667, 2.97626667, 2.97726667,
            2.98026667, 2.98126667, 2.98326667, 2.98426667, 2.98526667,
            2.98826667, 2.99026667, 2.99126667]),
        'states': np.array([
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        'expected_start_times': np.array([0.96106667, 1.96116667]),
        'expected_harp_times': np.array([3806874, 3806875])
    },
    {
        # contains 4 valid Harp clock barcodes and one invalid barcode
        'timestamps': np.array([
            0.14036667, 1.10146667, 1.10246667, 1.10346667, 1.10446667,
            1.11146667, 1.11246667, 1.11646667, 1.11746667, 1.11846667,
            1.11946667, 1.12146667, 1.12246667, 1.12546667, 1.12746667,
            1.12846667, 1.13046667, 1.13146667, 1.14046667, 2.10156667,
            2.10356667, 2.11156667, 2.11256667, 2.11656667, 2.11756667,
            2.11856667, 2.11956667, 2.12156667, 2.12256667, 2.12556667,
            2.12756667, 2.12856667, 2.13056667, 2.13156667, 2.14056667,
            3.10163333, 3.10263333, 3.11163333, 3.11263333, 3.11663333,
            3.11763333, 3.11863333, 3.11963333, 3.12163333, 3.12263333,
            3.12563333, 3.12763333, 3.12863333, 3.13063333, 3.13163333,
            3.14063333, 4.10173333, 4.11073333, 4.11173333, 4.11673333,
            4.11873333, 4.11973333, 4.12173333, 4.12273333, 4.12573333,
            4.12773333, 4.12873333, 4.13073333, 4.13173333, 4.14073333,
            5.1018    , 5.1028    , 5.1038    , 5.1108    , 5.1118    ,
            5.1168    , 5.1188    , 5.1198    , 5.1218    , 5.1228    ,
            5.1258    , 5.1278    , 5.1288    , 5.1308    , 5.1318    ,
            5.1368    , 5.1388    , 5.1398    , 5.14183333, 5.1428    ,
            5.14583333, 5.14783333, 5.14883333, 5.15083333, 5.15183333,
            5.16083333, 6.1059    ]),
        'states': np.array([
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 0]),
        'expected_start_times': np.array([1.10146667, 2.10156667, 3.10163333, 4.10173333]),
        'expected_harp_times': np.array([2600957, 2600958, 2600959, 2600960])
    }
]
# fmt: on


@mark.parametrize("test_input", testinput)
def test_create_reader(test_input):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_times, harp_times = decode_harp_clock(
            test_input["timestamps"],
            test_input["states"],
            baud_rate=1000,
        )

    assert np.allclose(start_times, test_input["expected_start_times"])
    assert np.allclose(harp_times, test_input["expected_harp_times"])

    # test alignment for samples 1/2 second after anchors
    ts = start_times + 0.5
    aligned_times = align_timestamps_to_anchor_points(ts, start_times, harp_times)
    assert np.allclose(aligned_times, harp_times + 0.5)

    # test alignment for samples 1/2 second before anchors
    ts = start_times - 0.5
    aligned_times = align_timestamps_to_anchor_points(ts, start_times, harp_times)
    assert np.allclose(aligned_times, harp_times - 0.5)
