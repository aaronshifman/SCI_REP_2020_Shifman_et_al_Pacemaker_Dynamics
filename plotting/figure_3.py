from pathlib import Path
from typing import Tuple, List, Optional

import figurefirst as fifi
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.io import loadmat

from model_code import ENA_DATA_PREFIX
from plotting import FIGURE_DATA
from plotting.plot_settings import save_fifi, init_fifi


def do_fft(v: np.ndarray, dt=1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fft.

    :param v: Membrane potential trace
    :param dt: Time step
    :return: FFT frequency, FFT power
    """
    v -= v.mean()  # Mean shift to remove DC power
    n = len(v)
    yf = fft(v)
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), int(n / 2))

    p = 2.0 / n * np.abs(yf[:n // 2])

    return xf, p


def should_keep(file_ix: int, folder_ix: int, xf: np.ndarray, p: np.ndarray) -> bool:
    """Filter function for recording validity.

    :param file_ix: File index
    :param folder_ix: Folder index
    :param xf: FFT frequency
    :param p: FFT power
    :return: If file should be analyzed
    """
    # these files are noisy and manually removed
    if (file_ix == 45 or file_ix == 37) and folder_ix == 1:
        return False

    # check spectral rules
    err = np.abs(xf - 60)
    p60 = p[err == np.min(err)]

    SNR = np.max(p) / p60

    harms = np.arange(1, 10) * 60
    val = np.any(np.abs(harms - xf[np.argmax(p)]) <= 5)

    # SNR too low
    if SNR < 1.5 or val:
        return False

    # Frequency too low
    if xf[np.argmax(p)] < 10:
        return False

    return True


def frequency_time_series(files: List[Path], folder_ix) -> np.ndarray:
    """Compute the time series of the crash from individual recordings.

    :param files:
    :param folder_ix:
    :return: Frequency time series
    """
    freqs = np.zeros((len(files),))
    # Iterate each file and if frequency is valid save frequency else use nan
    for file_ix, file in enumerate(files):
        v = loadmat(file)['recording'].flatten()
        xf, p = do_fft(v)
        f = xf[np.argmax(p)] if should_keep(file_ix, folder_ix, xf, p) else np.nan
        freqs[file_ix] = f

    return freqs


def experimental_bifurcation(t, folder_ix, folder, recording_interval=20) -> Optional[np.ndarray]:
    """Compute the experimental bifurcation diagram.

    If the time series never crashes return None
    :param t: Normalized time to interpolate
    :param folder_ix: Index of the folder
    :param folder: Path of the folder
    :param recording_interval: Time step between recordings (second)
    :return: Experimental bifurcation data for each recording
    """
    files = folder.rglob("*.mat")
    files = sorted(files, key=lambda x: int(Path(x).stem.split('o')[-1]))
    files = list(files)
    freqs = frequency_time_series(files, folder_ix)

    time = np.arange(len(files)) * recording_interval

    bad_times = np.where(np.isnan(freqs))[0]

    # nan freqs with valid freqs after i.e. a blip not the crash
    should_interpolate = bad_times[[np.any(~np.isnan(freqs[b:])) for b in bad_times]]
    mask = np.ones(freqs.shape, dtype=bool)
    mask[should_interpolate] = False

    # delete masked entries from freq and time
    time = time[mask]
    freqs = freqs[mask]
    freqs[np.isnan(freqs)] = 0

    if np.any(freqs == 0):
        crash_ix = np.where(freqs == 0)[0][0]  # first zero frequency
        time = time / time[crash_ix]  # normalize time to creash

        valids = np.where(time < 1)  # only store  pre-crash
        return interp1d(time[valids], freqs[valids], fill_value=np.nan, bounds_error=False)(t)


def compute_bifn_eqll(ax) -> None:
    """Extract bifurcation diagram for steady-state and orbit.

    :param ax: Axis to plot on
    :return: None
    """
    bifn = pd.read_csv(FIGURE_DATA / 'best_brown_target_orbit.dat', names=['ena', 'vmax', 'vmin', 'p1', 'p2', 'p3'],
                       delimiter=' ')
    bifn.ena *= 1e3
    bifn.vmax *= 1e3
    bifn.vmin *= 1e3
    bifn['lc'] = bifn.p2 == 2

    s_lc = 3
    us_fp = 2
    s_fp = 1

    bifurcation_set = bifn[~(bifn.lc) & (bifn.p1 == s_fp)]
    bifurcation_set.sort_values('ena')
    ax.plot(bifurcation_set.ena.values.tolist(), bifurcation_set.vmin, 'r-')
    ax.plot(bifurcation_set.ena.values.tolist(), bifurcation_set.vmax, 'r-')

    bif_e = bifurcation_set.iloc[-1]['ena']
    bif_v = bifurcation_set.iloc[-1]['vmax']

    print('e', bif_e, 'v', bif_v)

    bifurcation_set = bifn[(bifn.lc) & (bifn.p1 == s_lc)]
    bifurcation_set.sort_values('ena')
    ax.plot([bif_e] + bifurcation_set.ena.values.tolist(), [bif_v] + bifurcation_set.vmin.values.tolist(), 'g')
    ax.plot([bif_e] + bifurcation_set.ena.values.tolist(), [bif_v] + bifurcation_set.vmax.values.tolist(), 'g')

    bifurcation_set = bifn[~(bifn.lc) & (bifn.p1 == us_fp)]
    bifurcation_set.sort_values('ena')
    bifurcation_set = bifurcation_set[1:]
    ax.plot([bif_e] + bifurcation_set.ena.values.tolist(), [bif_v] + bifurcation_set.vmin.values.tolist(), 'k')


def compute_bifn_freq(ax, n: str, canonical=False) -> None:
    """Extract bifurcation diagram for frequency.

    :param ax: Axis to plot on
    :param n: Which name to extract
    :param canonical: If the model is the canonical model (plot thick)
    :return: None
    """
    bifn = pd.read_csv(FIGURE_DATA / f'{n}_frequency.dat', names=['ena', 'f', 'fmin', 'p1', 'p2', 'p3'],
                       delimiter=' ')
    bifn = bifn[bifn.p1 == 3]
    bifn = bifn.sort_values('ena')
    bifn.ena *= 1e3

    alpha = 1 if canonical else 150 / 255
    ax.plot(bifn.ena, bifn.f, 'g-', alpha=alpha)
    ax.plot([-100, bifn.ena.iloc[0]], [0, 0], 'r-')
    ax.plot([bifn.ena.iloc[0], bifn.ena.iloc[0]], [0, bifn.f.iloc[0]], 'k', linestyle='dotted', linewidth=0.5)


def run(fig_name='f3'):
    """Draw figure 3.

    :param fig_name: Name of the data for figure 3.
    :return: None
    """
    layout = init_fifi(fig_name)

    """
    Orbit Diagram /w eqilibria
    """
    ax = layout.axes['ax_a']
    compute_bifn_eqll(ax)
    ax.set_xlim([20, -20])
    ax.set_ylabel('V$_m$ [mV]')
    fifi.mpl_functions.adjust_spines(ax, ['bottom', 'left'], yticks=[-80, -60, -40], xticks=[-20, 0, 20],
                                     direction='out',
                                     smart_bounds=True)

    """
    Frequency bifurcation diagram
    """
    ax = layout.axes['ax_b']
    compute_bifn_freq(ax, 'best_brown_cell21')
    compute_bifn_freq(ax, 'best_black_expt25_cell1_file016')
    compute_bifn_freq(ax, 'best_black_expt28_cell1_file010')
    compute_bifn_freq(ax, 'best_brown_target', canonical=True)
    ax.set_xlim([20, -50])
    ax.set_ylim([0, 500])
    ax.set_ylabel('f [Hz]')
    ax.set_xlabel("E$_{Na}$ [mV]")
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], yticks=[0, 250, 500], xticks=[-50, 0, 20], direction='out',
                                     smart_bounds=True)

    """
    Experimental bifurcation diagram
    """
    experiment_folders = list(ENA_DATA_PREFIX.iterdir())
    interp_time = np.linspace(0, 1.5, 100)
    crashes = [experimental_bifurcation(interp_time, ix, folder) for ix, folder in enumerate(experiment_folders)]
    crashes = np.vstack(crashes).T
    ax = layout.axes['ax_c']
    ax.plot(interp_time, crashes, color=np.array([170, 170, 170]) / 255)
    ax.plot(interp_time, np.mean(crashes, axis=1), 'g', linewidth=2)
    ax.plot([1, 1.5], [0, 0], 'r', linewidth=2)
    ax.set_ylabel('f [Hz]')
    ax.set_xlabel('T$_{rel}$')
    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'], yticks=[0, 300, 600], xticks=[0, 1, 1.5], direction='out',
                                     smart_bounds=True)

    save_fifi(layout, fig_name)


if __name__ == "__main__":
    run()
