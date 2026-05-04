import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def apply_custom_plotting_flavor():
    """
    Applies the custom plotting style inspired by the supervisor's notebook.
    Features thin lines, grey ticks, LaTeX text formatting, and no top/right spines.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['font.size'] = 8
    
    # NOTE: Set to False if you don't have a local LaTeX installation
    plt.rcParams['text.usetex'] = False

    plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['figure.titlesize'] = 9

    # Axis and edge coloring (sleek grey instead of black)
    plt.rcParams['axes.edgecolor'] = '#8C8C8C'
    plt.rcParams['axes.labelcolor'] = '#595959'
    plt.rcParams['axes.linewidth'] = 0.4
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.titlesize'] = 9
    
    # Tick formatting
    plt.rcParams['xtick.color'] = '#8C8C8C'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['xtick.major.pad'] = 3
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 0.4
    plt.rcParams['xtick.minor.pad'] = 3
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['xtick.minor.width'] = 0.4
    
    plt.rcParams['ytick.color'] = '#8C8C8C'
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['ytick.major.pad'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 0.4
    plt.rcParams['ytick.minor.pad'] = 3
    plt.rcParams['ytick.minor.size'] = 1.5
    plt.rcParams['ytick.minor.width'] = 0.4

    # Grid formatting
    plt.rcParams['grid.color'] = '#BFBFBF'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linewidth'] = 0.2

    # Line formatting
    plt.rcParams['lines.linewidth'] = 0.6