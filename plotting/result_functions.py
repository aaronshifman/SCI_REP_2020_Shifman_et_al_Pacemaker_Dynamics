"""Helper functions for working with simulation data or analysis."""
import os
from pickle import load, dump
from typing import NamedTuple, Optional, Union, List, Tuple, Callable, Dict

import numpy as np
from scipy.interpolate import UnivariateSpline

from plotting import FIGURE_DATA

CHANGE_DATA = {
    'na': {'baseline': -10.1 / 65.5, 'amplitude': -9 / 30, 'slew_up': -82.4 / 179.5, 'slew_down': -41.6 / 102.7,
           'width': 0.189 / 0.331, 'frequency': -0.3},
    'k': {'baseline': 8.7 / 61.9, 'amplitude': -15.3 / 27.9, 'slew_up': -102.2 / 141.3, 'slew_down': -45.9 / 75.3,
          'width': 0.282 / 0.400, 'frequency': 0.2}}  # Smith and Zakon 2000


class MetricTuple(NamedTuple):
    """Data struct for extracted metrics."""

    amplitude: Optional[Union[float, List[float]]]
    slew_up: Optional[Union[float, List[float]]]
    slew_down: Optional[Union[float, List[float]]]


def create_fig_data(name, create_fcn: Callable) -> Dict:
    """Create (or load existing) figure data.

    :param name: Name of the figure data
    :param create_fcn: Function to call to create data if not exists
    :return: Data
    """
    data_path = FIGURE_DATA / f'{name}.pkl'
    if not os.path.exists(data_path):
        save_data = create_fcn()
        with open(data_path, 'wb') as f:
            dump(save_data, f)

    with open(data_path, 'rb') as f:
        return load(f)


def extract_measure(t, v) -> MetricTuple:
    """Extract amplitude and slew rates from a trace.

    :param t: Time
    :param v: Single membrane potential trace
    :return: Measures
    """
    a = max(v) - min(v)
    if a > 1:  # If there is an oscillation
        max_slew = np.max(np.diff(v)) / (t[1] - t[0])
        min_slew = -np.min(np.diff(v)) / (t[1] - t[0])
        amp = np.max(v) - np.min(v)
        return MetricTuple(amplitude=amp, slew_up=max_slew, slew_down=min_slew)

    return MetricTuple(amplitude=None, slew_up=None, slew_down=None)


def compute_metrics(time: np.ndarray, voltage: np.ndarray) -> MetricTuple:
    """Clip model trace and extract metrics for all block levels.

    :param time: Time series
    :param voltage: Membrane potential of all block levels
    :return: Named tuple with amplitude, slew_up and slew_down fields (array not float)
    """
    ix_half = int(0.9 * voltage.shape[1])  # clip transient
    metrics = zip(*(extract_measure(time[ix_half:], v[ix_half:]) for v in voltage))

    return MetricTuple(*metrics)


def map_metric(block_level, data: MetricTuple, metric: str, ion: str) -> Tuple[Optional[float], Optional[float]]:
    """Compute the block_level required to present the same percentage change as in Smith and Zakon 2000.

    :param block_level: Sequence of block levels
    :param data: Listed MetricTuple
    :param metric: Name of the metric to map
    :param ion: Name of the ion (Na/K)
    :return: Effective block level and metric value
    """
    data = np.array(data.__getattribute__(metric), dtype=float)  # extract metric from named tuple

    block_level = block_level[::-1]
    data = data[::-1]  # fitting requires increasing 'x'

    # clip out crash
    invalid = np.where(np.isnan(data))[0][0]
    data_trim = data[:invalid]
    trimed_block = block_level[:invalid]

    percent_change = (data_trim - data_trim[0]) / abs(data_trim[0])

    # root finding
    y_to_find = CHANGE_DATA[ion][metric]
    freduced = UnivariateSpline(trimed_block, percent_change - y_to_find, s=0)

    effective_block = freduced.roots()
    if effective_block:
        f = UnivariateSpline(trimed_block, data_trim, s=0)
        return effective_block, f(effective_block)
    return np.nan, np.nan
