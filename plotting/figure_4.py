from typing import Dict

import figurefirst as fifi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmax

from model_code.stock_model import *
from plotting.plot_settings import truncate_colormap, init_fifi, save_fifi
from plotting.result_functions import compute_metrics, map_metric, MetricTuple, create_fig_data

num_neurons = 100
block_level = np.arange(num_neurons)[::-1] / num_neurons


def create_data() -> Dict:
    """Create data for figure.

    :return: Computed data
    """
    voltages = {}
    metrics = {}
    for ion in ['Na', 'K']:
        params, group = set_model(name='best_brown_target', num_neurons=num_neurons)

        # brian string input doesn't like mS
        # i / num_neurons creates a sequence 0, 1/num_neurons, 2/num_neurons, ... to sweep block level
        conductance_str = '*'.join(str(params[f'g{ion}']).split()).replace('mS', 'msiemens') + ' * i / num_neurons'

        # update appropriate ion conductance with new g_ion
        group.__setattr__(f'g{ion}', conductance_str)
        s_mon = b2.StateMonitor(group, 'v', record=True, dt=0.001 * b2.ms)
        b2.run(200 * b2.ms, report='text')  # run simulation for 100 cells with varying block level
        metrics[f'metrics_{ion}'] = compute_metrics(s_mon.t / b2.ms, s_mon.v / b2.mV)
        voltages[f'v_{ion}'] = s_mon.v / b2.mV

    data = {**metrics, **voltages}
    return data


def make_hacked_colorbar(c, layout, ix) -> None:
    """Force a colorbar onto a new axis.

    :param c: Colormap to use
    :param layout: fifi layout object
    :param ix: axis index
    :return: None
    """
    # make colorbars
    # make a temporary figure plot a full color scale image
    # this is a hack to get a matching colorbar for the clipped colorscale

    legend_axis = layout.axes['legend_top_' + str(ix + 1)]
    plt.figure('_tmp')
    img = plt.imshow([[0, 0.7], [0, 0.7]], cmap=c)
    plt.close('_tmp')
    cbar = plt.colorbar(img, cax=legend_axis, ticks=[0, 0.7])
    cbar.ax.tick_params(size=0)
    cbar.outline.set_edgecolor('None')

    if ix == 0:
        cbar.ax.set_yticklabels([])
    else:
        cbar.ax.set_ylabel('Block Level', fontsize=10)


def run(fig_name='f4'):
    """Draw figure 4.

    :param fig_name: Name of the data for figure 4.
    :return: None
    """
    data = create_fig_data(fig_name, create_data)
    layout = init_fifi(fig_name)

    data['metrics_Na'] = MetricTuple(*data['metrics_Na'])
    data['metrics_K'] = MetricTuple(*data['metrics_K'])

    cs = ['Blues_r', 'Oranges_r']
    cmap = [truncate_colormap(matplotlib.cm.get_cmap(c), maxval=0.7, n=8) for c in cs]

    """
    Serial Block Snapshots
    """
    # use block levels 29, 39, ...
    # since ix goes 0, 99 we use 29 - 99 instead of 30-100
    # reverse since block 99 in the code is no block
    block_taken = np.arange(29, 100, 10)[::-1]

    for ix, ch in enumerate(['Na', 'K']):
        voltage_data = data[f'v_{ch}']
        colors = cmap[ix](np.linspace(0, 1, len(block_taken)))
        ax = layout.axes['ax_' + ch]
        for iy, b in enumerate(block_taken):
            block_data = voltage_data[b][182000:]  # cut off the first chunk to make nice plots
            pks = argrelmax(block_data)[0]
            if len(pks) == 0:
                to_plot = block_data[:6000]
            else:
                to_plot = block_data[pks[1]:]
                to_plot = to_plot[1500:7500]  # shift so not starting on peak
            ax.plot(to_plot, c=colors[iy, :-1])

        ax.set_ylim([-80, -40])
        if ix == 0:
            ax.set_ylabel('V$_m$ [mV]')
            fifi.mpl_functions.adjust_spines(ax, ['left'], yticks=[-80, -70, -60, -50, -40], direction='out')
        else:
            fifi.mpl_functions.adjust_spines(ax, ['none'], direction='out')

        make_hacked_colorbar(cmap[ix], layout, ix)

    """
    Block Metric Figures
    """
    na_blocks = []
    k_blocks = []
    for ix, metric in enumerate(['slew_up', 'slew_down', 'amplitude']):
        ax = layout.axes[metric]
        h1 = ax.plot(block_level, data['metrics_Na'].__getattribute__(metric), color=cmap[0](0.25))[0]
        h2 = ax.plot(block_level, data['metrics_K'].__getattribute__(metric), color=cmap[1](0.25))[0]
        x_block, y_block = map_metric(block_level, data['metrics_Na'], metric, 'na')
        ax.plot(x_block, y_block, 'o', color=cmap[0](0.25), markersize=4.5)
        na_blocks.append(x_block)  # block is flipped with 0 block being 0 conductance
        x_block, y_block = map_metric(block_level, data['metrics_K'], metric, 'k')
        ax.plot(x_block, y_block, 'o', color=cmap[1](0.25), markersize=4.5)
        k_blocks.append(x_block)

        if metric == 'slew_up':
            fifi.mpl_functions.adjust_spines(ax, ['bottom', 'left'], yticks=[0, 25, 50], xticks=[0, 0.35, 0.7],
                                             direction='out', smart_bounds=True)
            ax.set_ylabel('dV/dt [mV/ms]')
            ax.set_xlabel('Block Level')

        elif metric == 'slew_down':
            fifi.mpl_functions.adjust_spines(ax, ['bottom', 'left'], yticks=[0, 25, 50], xticks=[0, 0.35, 0.7],
                                             direction='out', smart_bounds=True)
            ax.set_yticklabels([])
            ax.set_xlabel('Block Level')

        elif metric == 'amplitude':
            fifi.mpl_functions.adjust_spines(ax, ['bottom', 'left'],
                                             xticks=[0, 0.35, 0.7], direction='out', smart_bounds=True)
            ax.set_xlabel('Block Level')
            ax.set_ylabel('V$_m$ [mV]')

        for item in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(10)

        ax.set_xlim([0, 0.7])

    print(f"Na: {np.nanstd(na_blocks) / np.nanmean(na_blocks)}")
    print(f"K: {np.nanstd(k_blocks) / np.nanmean(k_blocks)}")

    # set legend
    legend_axis = layout.axes['legend_bottom']
    legend_axis.legend([h1, h2], ['gNa', 'gK'], ncol=2)
    fifi.mpl_functions.adjust_spines(legend_axis, 'none')

    save_fifi(layout, fig_name)


if __name__ == "__main__":
    run()
