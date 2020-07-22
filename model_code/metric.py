"""RMSMetric class and helpers for model fitting."""
from typing import Tuple, Optional

import brian2modelfitting as bf
import numpy as np
from scipy.signal import argrelmax


class AlignedRMSMetric(bf.TraceMetric):
    """Brian2 metric for computing RMS error between model and target waveform.

    Overwrites get_features for aligning waveforms and get_errors for combining all (1) feature.
    """

    def __init__(self, cutoff: int):
        """Set the waveform cutoff for history.

        :param cutoff: Number of points to cutoff
        """
        self.cutoff = cutoff  # how much of transient to throw out (ms)
        super().__init__()

    def get_features(self, model_traces: np.ndarray, data_traces: np.ndarray, dt: float) -> np.ndarray:
        """Extract features for each model instance.

        :param model_traces: Incoming solution to test GOF
        :param data_traces: Single waveform to fit to
        :param dt: Time step
        :return: Computed errors
        """
        model_traces = np.squeeze(model_traces)
        model_traces = model_traces[:, self.cutoff:]
        data_traces = data_traces[0][self.cutoff:]

        # throw out low amplitude (<10mV) solutions
        half_max = np.nanmax(model_traces, axis=1)
        half_min = np.nanmin(model_traces, axis=1)
        valid_index = (half_max - half_min) > 0.01  # 10 mV
        valid_index = np.squeeze(valid_index)

        errs = np.zeros([model_traces.shape[0], 1])
        errs[~valid_index] = 1e6 # set invalid solutions to high error

        for col in np.where(valid_index)[0]:
            *_, err = lineup_peak(model_traces[col, :], data_traces)
            errs[col] = err
        return errs

    def get_errors(self, features):
        """Compute error for each instance.

        :param features: List of features (1) for each instance
        :return: 1D error measure for each instance
        """
        return np.mean(features, axis=1)


def lineup_peak(model, data, length=2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Align model recording to data from first peak.

    :param model: Model waveform
    :param data: Real recording
    :param length: Number of data cycles to use
    :return: aligned model, aligned data, and RMSE
    """
    pk_model = argrelmax(model)[0]  # find peaks in the model
    if len(pk_model) > 3:  # if theres at least 3 peaks after transient removed (an oscillation?)
        pk_data = argrelmax(data, order=1000)[0]
        if length:
            cycle_data = data[pk_data[2]: pk_data[2 + length]]
        else:
            cycle_data = data[pk_data[2]:]
        try:
            cycle_model = 1000 * model[pk_model[1]:pk_model[1] + len(cycle_data)]
            err = cycle_model - cycle_data
            err = np.sqrt(np.mean(err ** 2))  # RMS
            return cycle_model, cycle_data, err
        except ValueError:  # not enough oscillations to fit - use a large error
            return None, None, 1e6
    else:  # no oscillation use large error
        return None, None, 1e6
