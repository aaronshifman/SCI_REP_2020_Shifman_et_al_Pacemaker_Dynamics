from typing import Dict

import figurefirst as fifi
import numpy as np
from scipy.signal import argrelmax

from model_code.data_funcs import get_recording
from model_code.metric import lineup_peak
from model_code.stock_model import *
from plotting import BROWN_COLOR, BLACK_COLOR
from plotting.plot_settings import MultiPlotAxisHandler, save_fifi, init_fifi
from plotting.result_functions import create_fig_data


def create_data() -> Dict:
    """Create data for figure.

    :return: Computed data
    """
    save_data = {}
    for name in RECORDING_NAMES:
        # load data and run simulation
        t, v, dt, cutoff = get_recording(name)
        params, group = set_model(name=f'best_{name}', num_neurons=1)
        s_mon = b2.StateMonitor(group, 'v', record=True, dt=0.001 * b2.ms)
        b2.run(200 * b2.ms, report='text')

        # extract model after transient and remove units
        model, data = s_mon.v[0, cutoff + 1:], v[0][cutoff + 1:]
        model /= b2.volt
        cycle_model, cycle_data, err = lineup_peak(model, data, length=None)
        cycle_model = cycle_model[1500:7500]  # arb. numbers for a nice plot
        cycle_data = cycle_data[1500:7500]

        save_data[name] = {'t': np.arange(len(cycle_data)) / 1000, 'model': cycle_model, 'data': cycle_data,
                           'rmse': err, 'nrmse': 100 * err / (max(data) - min(data))}

    return save_data


def run(fig_name='f1'):
    """Draw figure 1.

    :param fig_name: Name of the data for figure 1.
    :return: None
    """
    data = create_fig_data(fig_name, create_data)
    layout = init_fifi(fig_name)

    """
    Canonical Model Fit
    """
    ax = layout.axes['canonical']
    plot_data = data['brown_target']
    ax.plot(plot_data['t'], plot_data['data'], color=BROWN_COLOR, linewidth=3)
    ax.plot(plot_data['t'], plot_data['model'], color='k')
    ax.set_ylabel('V$_m$ [mV]', labelpad=0)
    fifi.mpl_functions.adjust_spines(ax, ['left'], direction='out', yticks=[-75, -45], smart_bounds=True)
    ax.set_xlim([0, 6])

    """
    Normalized Data Overlay
    """
    ax = layout.axes['data_all']
    for name in RECORDING_NAMES:
        t, recording_voltage, _, cutoff = get_recording(name)
        signal = recording_voltage[0][cutoff + 1:]
        cycle_ix = argrelmax(signal, order=1000)[0][1:3]
        isi = (np.diff(cycle_ix) // 2)[0]
        cycle = signal[cycle_ix[1] - isi:cycle_ix[1] + isi]
        cycle = (cycle - np.min(cycle)) / (np.max(cycle) - np.min(cycle))

        t_norm = np.linspace(0, 1, len(cycle))
        color = BLACK_COLOR if 'black' in name else BROWN_COLOR
        zorder = 1000 if name == RECORDING_NAMES[2] else None # ordering hacking to make plot nicer
        ax.plot(t_norm, cycle, color=color, zorder=zorder, linewidth=2.25)

    ax.set_ylabel('V$_{rel}$', rotation=0, labelpad=0)
    ax.set_xlabel('T$_{rel}$')
    fifi.mpl_functions.adjust_spines(ax, 'none', direction='out')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    """
    All Model Fits
    """
    model_lines = []
    data_lines = []
    for ix, name in enumerate(RECORDING_NAMES):
        ax = layout.axes[name]
        color = BLACK_COLOR if 'black' in name else BROWN_COLOR
        plot_data = data[name]
        h1 = ax.plot(plot_data['t'], plot_data['data'], color=color, linewidth=3)[0]
        h0 = ax.plot(plot_data['t'], plot_data['model'], color='k')[0]

        fifi.mpl_functions.adjust_spines(ax, ['left'], direction='out', yticks=[-80, -20], smart_bounds=True)
        if ix == 0 or ix == 3:
            model_lines.append(h0)
            data_lines.append(h1)
            ax.set_ylabel('V$_m$ [mV]', labelpad=0)
        else:
            ax.set_yticklabels([])

        ax.set_xlim([0, 6])

        print({name: plot_data['nrmse']})
    legend_axis = layout.axes['legend']
    legend_axis.legend([tuple(model_lines), tuple(data_lines)], ['Model', 'Data'],
                       handler_map={tuple: MultiPlotAxisHandler()})
    fifi.mpl_functions.adjust_spines(legend_axis, 'none')

    save_fifi(layout, fig_name)


if __name__ == "__main__":
    run()
