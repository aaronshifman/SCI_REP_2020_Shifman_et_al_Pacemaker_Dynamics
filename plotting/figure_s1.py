from typing import Dict, Any

import figurefirst as fifi
import numpy as np
from numpy import exp # required for eval - do not remove

from model_code.stock_model import *
from plotting.plot_settings import init_fifi, save_fifi

def get_executable_functions() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Convert model gating functions into python executable functions.

    :return: tau_x function, x_inf function
    """
    model = get_generic_model().split('\n')
    tau_regex = re.compile('^tau.*')
    tau_fcn = [s for s in model if re.match(tau_regex, s)]
    tau_fcn = {f[4]: f.split('=')[1].split(":")[0] for f in tau_fcn}
    inf_regex = re.compile('^._inf')
    inf_fcn = [s for s in model if re.match(inf_regex, s)]
    inf_fcn = {f[0]: f.split('=')[1].split(":")[0] for f in inf_fcn}

    return tau_fcn, inf_fcn


def compute_tau(tau_fcn: Dict[str, str], params: Dict[str, Any], v) -> Dict[str, np.ndarray]:
    """Convert evaluate tau_x function strings.

    :param tau_fcn: Dictionary of tau functions
    :param params: Model parameter dictionary
    :param v: Voltage (Used for eval namespace)
    :return: tau_x(v)
    """
    tau_eval = {}
    param_names = ['scaling', 'theta', 'sigma1', 'sigma2']
    for ion, fcn in tau_fcn.items():
        for param in param_names:
            n = f'{param}_tau_{ion}'
            fcn = fcn.replace(n, str(params[n].base))

        tau_eval[ion] = eval(fcn) * 1000
    return tau_eval


def compute_inf(inf_fcn: Dict[str, str], params: Dict[str, Any], v) -> Dict[str, np.ndarray]:
    """Convert evaluate tau_x function strings.

    :param inf_fcn: Dictionary of tau functions
    :param params: Model parameter dictionary
    :param v: Voltage (Used for eval namespace)
    :return: x_inf(v)
    """
    inf_eval = {}
    param_names = ['theta', 'sigma']
    for ion, fcn in inf_fcn.items():
        for param in param_names:
            n = f'{param}_{ion}_inf'
            fcn = fcn.replace(n, str(params[n].base))

        inf_eval[ion] = eval(fcn)
    return inf_eval


def run(fig_name='fs1'):
    """Draw figure S1.

    :param fig_name: Name of the data for figure S1.
    :return: None
    """
    layout = init_fifi(fig_name)

    hs = []
    tau_fcn, inf_fcn = get_executable_functions()
    vars = {'na': {'act': 'm', 'inact': 'h'},
            'k': {"act": 'n', 'inact': 'q'},
            'ca': {'act': 'b', 'inact': 'g'}}

    colors = {"act": 'k', 'inact': 'r'}
    v = np.linspace(-80, -20, 300) / 1000
    for ix, name in enumerate(RECORDING_NAMES):
        width = 3 if ix == 0 else 1
        style = '-' if ix == 0 else '--'
        params = get_saved_params(name=f'best_{name}')

        tau_eval = compute_tau(tau_fcn, params, v)
        inf_eval = compute_inf(inf_fcn, params, v)

        for ion, var in vars.items():
            ax = layout.axes[f'gating_{ion}']
            for t, term in var.items():
                hs.append(ax.plot(v * 1000, inf_eval[term], colors[t], linewidth=width, linestyle=style)[0])

                fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], smart_bounds=True, yticks=[0, 0.5, 1],
                                                 xticks=[-80, -60, -40, -20], direction='out')

                if ion == 'ca':
                    ax.set_xlabel('V$_m$ [mV]')
                else:
                    ax.set_xticklabels([])
            ax.set_ylabel('x$_\infty$')

            ax = layout.axes[f'tau_{ion}']
            for t, term in var.items():
                ax.plot(v * 1000, tau_eval[term], colors[t], linewidth=width, linestyle=style)

            fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], smart_bounds=True, yticks=[0, 3, 6],
                                             xticks=[-80, -60, -40, -20], direction='out')
            if ion == 'ca':
                ax.set_xlabel('V$_m$ [mV]')
            else:
                ax.set_xticklabels([])

            ax.set_ylabel(f'$\\tau_x$ [ms]')

    legend_ax = layout.axes['legend']
    legend_ax.legend(hs[:2], ['Activation', 'Inactivation'])
    fifi.mpl_functions.adjust_spines(legend_ax, 'none')

    save_fifi(layout, fig_name)


if __name__ == "__main__":
    run()
