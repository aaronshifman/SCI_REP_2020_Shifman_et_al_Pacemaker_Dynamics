from typing import Dict

import figurefirst as fifi
import numpy as np
from scipy.signal import argrelmax

from model_code.data_funcs import get_recording
from model_code.stock_model import *
from plotting import BROWN_COLOR
from plotting.plot_settings import save_fifi, init_fifi
from plotting.result_functions import create_fig_data


def create_data() -> Dict:
    """Create data for figure.

    :return: Computed data
    """
    t, v, _, cutoff = get_recording('brown_target')
    params, group = set_model(name=f'best_brown_target', num_neurons=2)
    group.gCaT[1] = 0 * b2.msiemens  # [gCaT, gCaT] -> [gCaT, 0] i.e. pattern is [on, off]
    s_mon = b2.StateMonitor(group, ['v', 'I_Na', 'I_K', 'I_Ca', 'I_leak'], record=True, dt=0.001 * b2.ms)
    b2.run(200 * b2.ms, report='text')

    data = {}
    for ix, ca_state in enumerate(['on', 'off']):  # on off
        data[ca_state] = {}
        data[ca_state]['data'] = v[0][cutoff + 1:]
        for attr in ['v', 'I_Na', 'I_K', 'I_Ca', 'I_leak']:
            data[ca_state][attr] = s_mon.__getattr__(attr)[ix, cutoff + 1:]  # pull out currents from monitor

    return data


def run(fig_name='f2'):
    """Draw figure 2.

    :param fig_name: Name of the data for figure 2.
    :return: None
    """
    data = create_fig_data(fig_name, create_data)
    layout = init_fifi(fig_name)

    """
    Compute Peaks in Data, Ca On Model and Ca Off Model
    For lining up and presenting
    """
    pk = argrelmax(data['on']['data'], order=1000)[0]  # data
    pk_data = np.arange(pk[2], len(data['on']['data']))

    pk = argrelmax(data['on']['v'], order=1000)[0]  # with I_Ca
    pk_model_on = np.arange(pk[2], pk[2] + len(pk_data))

    pk = argrelmax(data['off']['v'], order=1000)[0]  # without I_Ca
    pk_model_off = np.arange(pk[2], pk[2] + len(pk_data))

    pk_data = pk_data[1500:7500]  # pick nice times corresponding to 6ms
    pk_model_on = pk_model_on[1500:7500]
    pk_model_off = pk_model_off[1500:7500]

    t = np.arange(len(pk_data)) / 1000

    """Fit Results On/Off"""
    for ix, (title, bnds) in enumerate(
            zip(['on', 'off'],
                [pk_model_on, pk_model_off])
    ):
        ax = layout.axes[f'v_{title}']
        h1 = ax.plot(t, data[title]['data'][pk_data], color=BROWN_COLOR, linewidth=3)[0]
        h0 = ax.plot(t, data[title]['v'][bnds] / b2.mV, color='k')[0]

        if ix == 0:
            ax.set_ylabel('V$_m$ [mV]', labelpad=0)

            legend_ax = layout.axes['legend_top']
            legend_ax.legend([h0, h1], ['Model', 'Data'])
            fifi.mpl_functions.adjust_spines(legend_ax, 'none')
        else:
            ax.set_yticklabels([])

        ax.set_xlim([0, 6])
        fifi.mpl_functions.adjust_spines(ax, ['left'], direction='out', yticks=[-75, -45],
                                         smart_bounds=True)

    """
    Ionic Breakdown
    """
    ion_colors = [np.array([202, 0, 32]) / 255,  # nice colors for each ion
                  np.array([244, 165, 130]) / 255,
                  np.array([146, 197, 222]) / 255,
                  np.array([5, 113, 176]) / 255]

    hs = []  # plot handles for each trace
    for iy, (bnds, state) in enumerate(zip([pk_model_on, pk_model_off], ['on', 'off'])):
        ax = layout.axes['i_' + state]
        for ix, current in enumerate(['I_K', 'I_leak', 'I_Ca', 'I_Na']):
            i_ion = -data[state][current][bnds] / b2.mA
            hs.append(ax.plot(t, i_ion, c=ion_colors[ix])[0])

            if state == 'on' and current == 'I_Ca':  # hack to create inflated I_Ca trace
                i_ion = -10 * data['on']['I_Ca'][bnds] / b2.mA
                hs.append(ax.plot(t, i_ion, c=ion_colors[ix], linestyle='dashed', linewidth=1)[0])
        ax.set_xlim([0, 6])

        if iy == 0:
            ax.set_ylabel('I$_m$ [mA/cm$^2$]', labelpad=0)
        fifi.mpl_functions.adjust_spines(ax, ['left'], direction='out', yticks=[-0.3, 0, 0.2],
                                         smart_bounds=True)

        if iy == 1:
            ax.set_yticklabels([])

        ax.set_ylim([-0.3, 0.2])

        if state == 'on':
            legend_ax = layout.axes['legend_bottom']
            legend_ax.legend(hs, ['I$_{K}$', 'I$_{Leak}$', 'I$_{Ca}$', r'$I_{Ca}\times 10$', 'I$_{Na}$'])
            fifi.mpl_functions.adjust_spines(legend_ax, 'none')

    save_fifi(layout, fig_name)


if __name__ == "__main__":
    run()
