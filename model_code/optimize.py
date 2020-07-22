"""Script to get around memory leak issues in brian2modelfitting.

This should be called as a subprocess from the main fitting script.
"""
import sys

import brian2modelfitting as bf

from model_code.data_funcs import get_recording, save_parameters
from model_code.metric import AlignedRMSMetric
from model_code.stock_model import *


def run(repeat_number: int, model_name: str) -> None:
    """Run the nth (repeat_number) optimization on a given recording.

    :param repeat_number: Optimization repeat
    :param model_name: Name of the recording
    :return: None
    """
    tt_new, ptar_new, dt, cutoff = get_recording(model_name)

    model = get_generic_model()
    fitter = bf.TraceFitter(model=model, input_var='I', output_var='v', input=0 * ptar_new * b2.amp,
                            output=ptar_new,
                            dt=dt * b2.ms,
                            n_samples=5000,  # < 50000
                            method='euler',
                            param_init={'v': -50 * b2.mV})

    opt = bf.NevergradOptimizer()
    res, error = fitter.fit(callback="progressbar", n_rounds=3,
                            optimizer=opt,
                            metric=AlignedRMSMetric(cutoff + 1), **param_ranges)

    save_parameters(f'{model_name}_{repeat_number}', res)


if __name__ == '__main__':
    *_, ix, name = sys.argv
    run(ix, name)
