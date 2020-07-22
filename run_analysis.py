"""Main entry point for analysis.

Creates xpp files and LaTeX parameter tables
Runs all analysis and figure code
"""
from model_code import RECORDING_NAMES
from model_code.stock_model import create_xpp_model, create_data_table
from plotting import figure_1, figure_2, figure_3, figure_4, figure_s1


def generate_model_strings() -> None:
    """Create XPP and supplemental tables.

    :return: None
    """
    for n in RECORDING_NAMES:
        create_xpp_model(name=f'best_{n}')

    create_data_table()


if __name__ == "__main__":
    generate_model_strings()
    figure_1.run()
    figure_2.run()
    figure_3.run()
    figure_4.run()
    figure_s1.run()
