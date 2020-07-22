"""Model simulation and manipulation core functions."""

from pathlib import Path

RECORDING_NAMES = ['brown_target',  # MODEL i
                   'black_expt25_cell1_file016',  # MODEL ii
                   'brown_cell21',  # MODEL iii
                   'black_expt28_cell1_file010',  # MODEL iv
                   ]
DATA_PATH = Path('data')
MODEL_SAVE_PREFIX = DATA_PATH / 'model_solutions'
ENA_DATA_PREFIX = DATA_PATH / 'sodium_free_data'
RECORDING_PREFIX = DATA_PATH / 'recordings'
SUPPLEMENTAL_PATH = Path('supplemental')
MODEL_PATH = DATA_PATH / 'MODEL_PARAMETRIC'
N_FITS = 10
