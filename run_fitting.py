"""Main entry point for running the model fits.

Due to a memory leak this spawns a subprocess python instance to run each fit seperately
Saves all models and computes the best one.
"""
import subprocess
from collections import defaultdict
from itertools import product
from typing import Dict

import numpy as np
from brian2 import NeuronGroup

from model_code.data_funcs import get_recording, set_units, load_parameters, save_parameters
from model_code.metric import lineup_peak
from model_code.stock_model import *

PARAM_LIST = Dict[str, List[float]]


def save_best_parameters(name: str, parameters: PARAM_LIST, ix: int) -> None:
    """Save best parameter from all parameters.

    :param name: Of the recording
    :param parameters: Dictionary of all fits
    :param ix: Index of best parameter to save
    :return: None
    """
    best_struct = dict()
    # Iterate all parameters and extract best one
    for key, val in parameters.items():
        if key == 'name':  # skip name field
            continue
        best_struct[key] = val[ix]

    save_parameters(f"best_{name}", best_struct)


def load_all_parameters() -> PARAM_LIST:
    """Load all instances of parameters into a dictionary with each entry a list of all version of that parameter.

    :return: Bulk parameters (parameters[p] = 4 names * 10 instances)
    """
    parameters = defaultdict(list)
    # For each model fit add the parameter to the parameter set

    for name, ix in product(RECORDING_NAMES, range(N_FITS)):
        data = load_parameters(name, ix)
        for key, value in data.items():  # iterate of each parameter and append to list
            parameters[key] += [value]
        parameters['name'] += [name]

    return parameters


def construct_params_model(parameters: PARAM_LIST) -> NeuronGroup:
    """Construct a model network.

     Each cell is uncoupled and represents an instance of a fit set of parameters. In total there are
     n_recording*n_fits = 40

    :param parameters: Struct of parameters
    :param n: Number of repeats
    :return: Model group
    """
    # Make enough neurons to set all unique parameters
    _, group = set_model(name='brown_target_0', num_neurons=len(parameters['gNa']))

    # Update group parameters for each neuron
    for key, value in parameters.items():
        if key == 'name':  # skip name field
            continue
        group.__setattr__(key, set_units(key, value, big=True))
    return group


def run_all_fits() -> Tuple[np.ndarray, PARAM_LIST]:
    """Run models for all saved parameters.

    :return: None
    """
    parameters = load_all_parameters()
    neuron_group = construct_params_model(parameters)
    s_mon = b2.StateMonitor(neuron_group, 'v', record=True, dt=0.001 * b2.ms)
    b2.run(200 * b2.ms, report='text')

    return s_mon.v / b2.volt, parameters


def extract_best() -> None:
    """Extract the best fit for each recording and save it.

    :return: None
    """
    v, parameters = run_all_fits()
    for ix, name in enumerate(RECORDING_NAMES):
        tt_new, ptar_new, dt, cutoff = get_recording(name)
        data = ptar_new[0][cutoff + 1:]
        # map neuron name to their index
        nrn_ixs = np.where(np.array(parameters['name']) == name)[0]
        voltage = v[nrn_ixs, cutoff + 1:]
        errors = [lineup_peak(model, data)[-1] for model in voltage]

        best_ix = np.argmin(errors)
        save_best_parameters(name, parameters, nrn_ixs[best_ix])


def spawn_optimize_process(recording_name: str, repetition: int) -> None:
    """Create a new python process for the optimizer.

    There appears to be a memory leak in brian2modelfitter==0.3.0. In order to avoid this we spawn a new python process
    for each optimization. This forces the OS to cleanup memory.

    :param recording_name: Name of the recording to fit
    :param repetition: Repetition number
    :return: None
    """
    # run main.optimize with arguments repetition and recording_name
    process = subprocess.Popen(['python3', 'main/optimize.py', str(repetition), recording_name], stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    _ = process.poll()


if __name__ == "__main__":
    # run each of the files with 0-N_FITS index
    [spawn_optimize_process(n, ix) for n, ix in product(RECORDING_NAMES, range(N_FITS))]
    extract_best()
