"""Helper functions for plotting."""
import figurefirst as fifi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from figurefirst import FigureLayout
from matplotlib import patches as mpatches
from matplotlib.colors import Colormap

from plotting import FIGURE_PATH


def save_fifi(layout: FigureLayout, name: str) -> None:
    """Finalize fifi figure and save.

    Insert figure to panels layer and add enable the annotation layer.
    Save and close the figure.

    :param layout: Figurefirst layout
    :param name: Name of the figure
    :return: None
    """
    layout.insert_figures('panels', cleartarget=True)
    layout.set_layer_visibility('annotation', True)
    layout.write_svg(FIGURE_PATH / f'{name}.svg')
    plt.close('all')


def init_fifi(name: str) -> FigureLayout:
    """Initialize a figure first figure.

    :param name: Name of the figure
    :return: Figure layout to edit
    """
    layout = fifi.svg_to_axes.FigureLayout(FIGURE_PATH / f'{name}.svg')
    layout.make_mplfigures()
    layout.fig.set_facecolor('None')

    return layout


class MultiPlotAxisHandler:
    """Axis handler to create a 2-line axis entry sharing the same vertical space."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox) -> mpatches.Rectangle:
        """Draw the legend entry given the handle.

        :param legend:
        :param orig_handle:
        :param fontsize:
        :param handlebox:
        :return: Legend pach
        """
        x0, y0 = handlebox.xdescent, handlebox.ydescent

        width, full_height = handlebox.width, handlebox.height
        height = orig_handle[0]._linewidth
        patch = mpatches.Rectangle([x0, y0 + full_height / 2 - height / 2], width / 2, height,
                                   facecolor=orig_handle[0]._color,
                                   edgecolor=None, hatch=None, lw=orig_handle[0]._linewidth,
                                   transform=handlebox.get_transform())

        height = orig_handle[1]._linewidth

        patch2 = mpatches.Rectangle([x0 + width / 2.01, y0 + full_height / 2 - height / 2], width / 2, height,
                                    facecolor=orig_handle[1]._color,
                                    edgecolor=None, hatch=None, lw=orig_handle[1]._linewidth,
                                    transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        handlebox.add_artist(patch2)
        return patch


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100) -> Colormap:
    """Take a colormap and cut it so color saturates within bounds.

    :param cmap: Colormap to cut
    :param minval: Minimum color value
    :param maxval: Maximum color value
    :param n: Number of segments
    :return: Truncated color map
    """
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
