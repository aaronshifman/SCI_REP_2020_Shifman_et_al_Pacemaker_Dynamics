"""Common collection of solution IO / unit manipulation tools."""
from pickle import load, dump
from typing import Dict

import brian2 as b2
import numpy as np
import pandas as pd

from model_code import RECORDING_PREFIX, MODEL_SAVE_PREFIX


def load_parameters(name: str, ix: int) -> Dict[str, tuple]:
    """Load saved parameters with integer suffix.

    :param name: Name of the model file to load
    :param ix: Index of the save file [0, n)
    :return: Parameters from file
    """
    with open(MODEL_SAVE_PREFIX / f"{name}_{ix}.pkl", 'rb') as f:
        parameters = load(f)

    return parameters


def get_recording(name: str, dt=0.001, t_hist=50):
    """Get recording data/metadata given the name.

    :param name: Name of the recording
    :param dt: Interpolated time step
    :param t_hist: Length of time to pad - this forms the cutoff
    :return: time, membrane potential, time step, cutoff (what time is the history function)
    """
    data = pd.read_csv(RECORDING_PREFIX / f"{name}.csv", names=['t', 'v'], skiprows=1)
    recording_time = data.t.values
    voltage = data.v.values

    # data originally MATLAB: strange dimension so flatten
    recording_time = np.array([recording_time.flatten()])
    voltage = np.array([voltage.flatten()])

    t_new = np.arange(-t_hist, np.max(recording_time), dt)
    cutoff = int(t_hist / dt)

    # interpolate with zeros for history
    interpolated_voltage = np.array([np.interp(t_new, recording_time[0], voltage[0], left=0)])

    # Implausible membrane potential shift down into reasonable bounds
    if name == "brown_cell21":
        interpolated_voltage -= 20
    if name == 'black_expt28_cell1_file010':
        interpolated_voltage -= 30

    return t_new, interpolated_voltage, dt, cutoff


def set_units(name: str, value: float, big=False) -> b2.Unit:
    """Convert numeric values into brian2 units.

    :param name: Name of the parameter
    :param value: Value of the parameter
    :param big: If parameter was scaled by 1/1000 (SI) re-inflate it
    :return: Value with units
    """
    if name[0] == 'g':
        mult = b2.msiemens  # conductances
    elif name[:2] == 'e_' or name[:6] == "theta_" or name[:5] == 'sigma':
        mult = b2.mvolt  # equilibrium potential / gating width / gating location
    elif name[:8] == "scaling_":
        mult = b2.msecond  # time constant
    else:
        raise ValueError('Unknown Unit Type')
    if big:
        mult = 1000 * mult
    return value * mult


def get_saved_params(name: str) -> Dict[str, b2.Unit]:
    """Load parameters from a file and set the units.

    :param name: Name of the parameter file
    :return: Parameters with units
    """
    with open(MODEL_SAVE_PREFIX / f'{name}.pkl', 'rb') as f:
        params = load(f)

    for k, v in params.items():
        params[k] = set_units(k, v, big=True)

    return params


def save_parameters(name: str, params: Dict[str, float]) -> None:
    """Save model parameters to file.

    :param name: Name of the file to save
    :param params: Parameters to save
    :return: None
    """
    with open(MODEL_SAVE_PREFIX / f"{name}.pkl", 'wb') as f:
        dump(params, f)
