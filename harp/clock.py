import numpy as np
import warnings


def align_timestamps_to_harp_clock(timestamps_to_align, start_times, harp_times):
    """
    Aligns a set of timestamps to the Harp clock.

    We assume these timestamps are acquired by a system that is
    not aligned to the Harp clock. This function finds the nearest
    anchor point in the Harp clock for each timestamp, and then
    interpolates between the anchor points to align the timestamps.

    `decode_harp_clock` must be run first, in order to find the
    start of each second in the Harp clock.

    Parameters
    ----------
    timestamps_to_align : np.array
        Local timestamps (in seconds) to align to the Harp clock
    start_times : np.array
        Local start times of each second in the Harp clock
        (output by `decode_harp_clock`)
    harp_times : np.array
        Harp clock times in seconds
        (output by `decode_harp_clock`)

    Returns
    -------
    aligned_times : np.array
        Aligned timestamps
    """

    if len(start_times) != len(harp_times):
        raise ValueError(
            "The number of start times must equal the number of Harp times"
        )

    N = len(start_times)

    slopes = np.zeros((N + 1,))
    offsets = np.zeros((N + 1,))

    # compute overall slope and offset
    A = np.vstack([start_times, np.ones(len(start_times))]).T
    slopes[0], offsets[0] = np.linalg.lstsq(A, harp_times, rcond=None)[0]

    # compute slope and offset for each segment
    for i in range(N):
        x = start_times[i : i + 2]
        y = harp_times[i : i + 2]
        A = np.vstack([x, np.ones(len(x))]).T
        slopes[i + 1], offsets[i + 1] = np.linalg.lstsq(A, y, rcond=None)[0]

    # find the nearest anchor point for each timestamp to align
    nearest = np.searchsorted(start_times, timestamps_to_align, side="left")

    nearest[nearest == N] = 0

    # interpolate between the anchor points
    aligned_times = timestamps_to_align * slopes[nearest] + offsets[nearest]

    return aligned_times


def decode_harp_clock(timestamps, states, baud_rate=1000.0):
    """
    Decodes Harp clock times (in seconds) from a sequence of local
    event timestamps and states.

    The Harp Behavior board can be configured to output a digital
    signal that encodes the current Harp time as a 32-bit integer,
    which is emitted once per second.

    The format of the signal is as follows:

      Default value: high
      Start bit: low -- indicates the transition to the next second
      8 bits: byte 0, with least significant bit first
      2 bits: high / low transition
      8 bits: byte 1, with least significant bit first
      2 bits: high / low transition
      8 bits: byte 2, with least significant bit first
      2 bits: high / low transition
      8 bits: byte 3, with least significant bit first
      Final bit: reset to high

    Although the baud rate of the internal Harp clock is
    100 kHz, the Behavior board outputs the clock signal
    at a lower baud rate (typically 1 kHz), so it can be
    acquired by data acquisition systems with sample rates
    as low as 5 kHz.

    Parameters
    ----------
    timestamps : np.array
        Float times in seconds for each clock line transition
        If the acquisition system outputs integer sample numbers
        for each event, divide by the sample rate to convert to seconds
    states : np.array
        States (1 or 0) for each clock line transition
    baud_rate : float
        The baud rate of the Harp clock signal

    Returns
    -------
    start_times : np.array
        Timestamps at which each second begins
    harp_times : np.array
        Harp clock times in seconds
    """

    min_delta = 0.5  # seconds -- Harp clock events must always be
    # at least this far apart

    barcode_edges = get_barcode_edges(timestamps, min_delta)

    start_times = np.array([timestamps[edges[0]] for edges in barcode_edges])

    harp_times = np.array(
        [
            convert_barcode(
                timestamps[edges[0] : edges[1]],
                states[edges[0] : edges[1]],
                baud_rate=baud_rate,
            )
            for edges in barcode_edges
        ]
    )

    start_times_corrected, harp_times_corrected = remove_outliers(
        start_times, harp_times
    )

    return start_times_corrected, harp_times_corrected


def get_barcode_edges(timestamps, min_delta):
    """
    Returns the start and end indices of each barcode

    Parameters
    ----------
    timestamps : np.array
        Timestamps (ins seconds) of clock line events
        (high and low states)
    min_delta : int
        The minimum length between the end of one
        barcode and the start of the next

    Returns
    -------
    edges : list of tuples
        Contains the start and end indices of each barcode
    """

    (splits,) = np.where(np.diff(timestamps) > min_delta)

    return list(zip(splits[:-1] + 1, splits[1:] + 1))


def convert_barcode(transition_times, states, baud_rate):
    """
    Converts Harp clock barcode to a clock time in seconds

    Parameters
    ----------
    transition_times : np.array
        Times (in seconds) each clock line transition
    states : np.array
        states (1 or 0) for each clock line transition
    baud_rate : float
        The baud rate of the clock signal

    Returns
    -------
    harp_time : int
        Harp time in seconds for the current barcode

    """

    intervals = np.round(np.diff(transition_times * baud_rate)).astype("int")

    barcode = np.concatenate(
        [np.ones((count,)) * state for state, count in zip(states[:-1], intervals)]
    ).astype("int")

    val = np.concatenate(
        (np.arange(1, 9), np.arange(11, 19), np.arange(21, 29), np.arange(31, 39))
    )

    s = np.flip(barcode[val])
    harp_time = s.dot(2 ** np.arange(s.size)[::-1])

    return harp_time


def remove_outliers(start_times, harp_times):
    """
    Removes outliers from the Harp clock times

    These outliers are caused by problems decoding
    the Harp clock signal, leading to consecutive times that
    do not increase by exactly 1. These will be removed from
    the array of Harp times, so they will not be used
    as anchor points during subsequent clock alignment.

    If the times jump to a new value and continue to
    increase by 1, either due to a reset of the Harp clock
    or a gap in the data, these will be ignored.

    Parameters
    ----------
    start_times : np.array
        Harp clock start times in seconds
    harp_times : np.array
        Harp clock times in seconds

    Returns
    -------
    corrected_start_times : np.array
        Corrected Harp clock times in seconds
    corrected_harp_times : np.array
        Corrected Harp clock times in seconds
    """

    original_indices = np.arange(len(harp_times))

    new_indices = np.concatenate(
        [
            sub_array
            for sub_array in np.split(
                original_indices, np.where(np.diff(harp_times) != 1)[0] + 1
            )
            if len(sub_array) > 1
        ]
    )

    num_outliers = len(original_indices) - len(new_indices)

    if num_outliers > 0:
        warnings.warn(
            f"{num_outliers} outlier{'s' if num_outliers > 1 else ''} "
            + "found in the decoded Harp clock. Removing..."
        )

    return start_times[new_indices], harp_times[new_indices]
