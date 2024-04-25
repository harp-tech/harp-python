import numpy as np
import warnings


def decode_harp_clock(sample_numbers, states, sample_rate=30000.0, baud_rate=1000.0):
    """
    Decodes Harp clock times (in seconds) from a sequence of sample
    numbers and states.

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
    sample_numbers : np.array
        Integer sample numbers for each clock line transition
    states : np.array
        States (1 or 0) for each clock line transition
    sample_rate : float
        The sample rate at which the clock signal was acquired
    baud_rate : float
        The baud rate of the clock signal

    Returns
    -------
    start_samples : np.array
        Sample numbers at which each second begins
    harp_times : np.array
        Harp clock times in seconds
    """

    min_delta = int(sample_rate / 2)  # 0.5 seconds

    barcode_edges = get_barcode_edges(sample_numbers, min_delta)

    start_samples = [sample_numbers[edges[0]] for edges in barcode_edges]

    harp_times = [
        convert_barcode(
            sample_numbers[edges[0] : edges[1]],
            states[edges[0] : edges[1]],
            sample_rate=sample_rate,
            baud_rate=baud_rate,
        )
        for edges in barcode_edges
    ]

    harp_times_corrected = correct_outliers(np.array(harp_times))

    return np.array(start_samples), harp_times_corrected


def get_barcode_edges(sample_numbers, min_delta):
    """
    Returns the start and end indices of each barcode

    Parameters
    ----------
    sample_numbers : np.array
        Sample numbers of clock line (high and low states)
    min_delta : int
        The minimum number of samples between each barcode

    Returns
    -------
    edges : list of tuples
        Contains the start and end indices of each barcode
    """

    (splits,) = np.where(np.diff(sample_numbers) > min_delta)

    return list(zip(splits[:-1] + 1, splits[1:] + 1))


def convert_barcode(sample_numbers, states, sample_rate, baud_rate):
    """
    Converts Harp clock barcode to a clock time in seconds

    Parameters
    ----------
    sample_numbers : np.array
        Integer sample numbers for each clock line transition
    states : np.array
        states (1 or 0) for each clock line transition
    sample_rate : float
        The sample rate at which the clock signal was acquired
    baud_rate : float
        The baud rate of the clock signal

    Returns
    -------
    harp_time : int
        Harp time in seconds for the current barcode

    """

    samples_per_bit = int(sample_rate / baud_rate)
    middle_sample = int(samples_per_bit / 2)

    intervals = np.diff(sample_numbers)

    barcode = np.concatenate(
        [np.ones((count,)) * state for state, count in zip(states[:-1], intervals)]
    ).astype("int")

    val = np.concatenate(
        [
            np.arange(
                samples_per_bit + middle_sample + samples_per_bit * 10 * i,
                samples_per_bit * 10 * i - middle_sample + samples_per_bit * 10,
                samples_per_bit,
            )
            for i in range(4)
        ]
    )
    s = np.flip(barcode[val])
    harp_time = s.dot(2 ** np.arange(s.size)[::-1])

    return harp_time


def correct_outliers(harp_times):
    """
    Corrects outliers in the Harp clock times

    Parameters
    ----------
    harp_times : np.array
        Harp clock times in seconds

    Returns
    -------
    corrected_harp_times : np.array
        Corrected Harp clock times in seconds
    """

    diffs = np.abs(np.diff(harp_times))

    # Find the indices of the outliers
    outliers = np.where(diffs > 1)[0]

    if len(outliers) == 1:
        warnings.warn("One outlier found in the decoded Harp clock. Correcting...")
    elif len(outliers) > 1:
        warnings.warn(
            f"{len(outliers)} outliers found in the decoded Harp clock. Correcting..."
        )

    # Correct the outliers
    for outlier in outliers:
        harp_times[outlier + 1] = harp_times[outlier] + 1

    return harp_times
