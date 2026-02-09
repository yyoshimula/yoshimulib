"""
Plotting utilities for paper/presentation figures
Python conversion from yMATLAB/utility/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Union, List
from datetime import datetime
import warnings


def fig4paper(asis: int = 1, fig: Optional[Figure] = None, n_fig: int = 1,
              content_type: str = 'vector', save: bool = True) -> None:
    """
    # Setting and saving figure for conference or journal manuscript

    Parameters
    ----------
    asis : int
        0 = resize figure based on layout, 1 = keep current size
    fig : Figure, optional
        matplotlib figure object (default: current figure)
    n_fig : int
        figure number for filename
    content_type : str
        'vector' for PDF/SVG, 'image' for PNG

    Returns
    -------
    None

    Notes
    -----
    論文用にfigureをいい感じに設定し，保存

    References
    ----------
    NA

    Revisions
    ---------
    20240823  y.yoshimura y.yoshimula@gmail.com, major update

    See also
    --------
    fig4presen
    """
    if fig is None:
        fig = plt.gcf()

    # Adjust font settings
    _adjust_font(fig)

    # Optimize figure appearance
    _optimize_fig(fig)

    if asis == 0:
        n_rows, n_cols = _detect_layout(fig)
        w_cm, h_cm = _decide_fig_size(n_rows, n_cols)
        # Convert cm to inches (matplotlib uses inches)
        fig.set_size_inches(w_cm / 2.54, h_cm / 2.54)

    # Apply LaTeX-style labels if using LaTeX backend
    apply_latex_labels(fig)

    # Export
    if save:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        if content_type == 'vector':
            fname = f'fig{n_fig}_{timestamp}.pdf'
            fig.savefig(fname, format='pdf', dpi=600, bbox_inches='tight',
                        transparent=True)
        else:
            fname = f'fig{n_fig}_{timestamp}.png'
            fig.savefig(fname, format='png', dpi=600, bbox_inches='tight',
                        transparent=True)


def _adjust_font(fig: Figure) -> None:
    """
    # Font settings for AIAA-style figures

    Parameters
    ----------
    fig : Figure
        matplotlib figure object

    Returns
    -------
    None
    """
    target_font_size = 16

    for ax in fig.axes:
        # Font settings
        ax.tick_params(labelsize=target_font_size)

        # Try to set Times New Roman, fall back to serif
        try:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontname('Times New Roman')
        except Exception:
            pass

        # Title font size
        ax.title.set_fontsize(target_font_size + 2)
        ax.title.set_fontweight('bold')

        # Axis label font size
        ax.xaxis.label.set_fontsize(target_font_size)
        ax.yaxis.label.set_fontsize(target_font_size)

        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(target_font_size)

    # Legend font settings
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(target_font_size - 4)


def _optimize_fig(fig: Figure) -> None:
    """
    # Optimize figure appearance

    Parameters
    ----------
    fig : Figure
        matplotlib figure object

    Returns
    -------
    None
    """
    for ax in fig.axes:
        # Line width adjustment
        for line in ax.get_lines():
            if line.get_linewidth() < 1.0:
                line.set_linewidth(1.0)

            # Marker size adjustment
            if line.get_markersize() > 0 and line.get_markersize() < 6:
                line.set_markersize(6)

        # Grid settings
        ax.grid(True, linestyle='-', alpha=0.3)

        # Axis settings
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


def apply_latex_labels(fig: Optional[Figure] = None) -> None:
    """
    # Apply LaTeX-style labels (MATLAB: applyLatexLabels)

    Parameters
    ----------
    fig : Figure, optional
        matplotlib figure object (default: current figure)
    """
    if fig is None:
        fig = plt.gcf()

    try:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    except Exception:
        # LaTeX not available, use mathtext
        plt.rc('text', usetex=False)
        plt.rc('mathtext', fontset='stix')
        plt.rc('font', family='STIXGeneral')

    # Force redraw of labels/titles to respect rc settings
    for ax in fig.axes:
        ax.set_title(ax.get_title())
        ax.set_xlabel(ax.get_xlabel())
        ax.set_ylabel(ax.get_ylabel())
        if hasattr(ax, 'zaxis'):
            ax.set_zlabel(ax.get_zlabel())

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_text(text.get_text())


def applyLatexLabels(fig: Optional[Figure] = None) -> None:
    """
    # MATLAB-compatible alias of apply_latex_labels
    """
    apply_latex_labels(fig)


def _detect_layout(fig: Figure) -> tuple:
    """
    # Detect subplot layout

    Parameters
    ----------
    fig : Figure
        matplotlib figure object

    Returns
    -------
    n_rows : int
        number of rows
    n_cols : int
        number of columns
    """
    axes = [ax for ax in fig.axes if not ax.get_label() == '<colorbar>']
    if len(axes) == 0:
        return 1, 1

    # Get positions
    positions = np.array([ax.get_position().bounds for ax in axes])

    # Group by y position (row)
    y_centers = np.round(positions[:, 1] * 100).astype(int)
    n_rows = len(np.unique(y_centers))

    # Group by x position (column)
    x_centers = np.round(positions[:, 0] * 100).astype(int)
    n_cols = len(np.unique(x_centers))

    return n_rows, n_cols


def _decide_fig_size(n_rows: int, n_cols: int) -> tuple:
    """
    # Decide figure size based on layout

    Parameters
    ----------
    n_rows : int
        number of rows
    n_cols : int
        number of columns

    Returns
    -------
    w_cm : float
        width in cm
    h_cm : float
        height in cm
    """
    # Width
    if n_cols == 1:
        w_cm = 8.3  # 1-column
    elif n_cols == 2 and n_rows == 1:
        w_cm = 17.0  # 2-column side by side
    else:
        w_cm = min(8.3 * n_cols * 0.9, 17.0)  # Max 17 cm

    # Height
    base_h = 8.3 * 0.68
    h_cm = base_h * n_rows * 0.95

    return w_cm, h_cm


def figs4paper(asis: int = 0, content_type: str = 'vector') -> None:
    """
    # Setting and saving all visible figures for manuscript

    Parameters
    ----------
    asis : int
        0 = resize figure based on layout, 1 = keep current size
    content_type : str
        'vector' for PDF/SVG, 'image' for PNG

    Returns
    -------
    None

    Notes
    -----
    表示されているfig全てを保存

    References
    ----------
    NA

    Revisions
    ---------
    20240823  y.yoshimura y.yoshimula@gmail.com, major update

    See also
    --------
    fig4paper
    """
    figs = [plt.figure(i) for i in plt.get_fignums()]

    for k, fig in enumerate(figs):
        # Check if already saved (using a custom attribute)
        if hasattr(fig, '_fig4papers_saved') and fig._fig4papers_saved:
            print(f'Skipped (already saved): Figure {k + 1}')
            continue

        fig4paper(asis, fig, k + 1, content_type)

        # Mark as saved
        fig._fig4papers_saved = True
        fig._fig4papers_timestamp = datetime.now()

        print(f'Saved: Figure {k + 1}')


def fig4presen(fig: Optional[Figure] = None, ax: Optional[Axes] = None) -> None:
    """
    # Setting and saving figure for presentation slide

    Parameters
    ----------
    fig : Figure, optional
        matplotlib figure object (default: current figure)
    ax : Axes, optional
        matplotlib axes object (default: current axes)

    Returns
    -------
    None

    Notes
    -----
    プレゼンテーション用にfigureをいい感じに設定し，保存

    References
    ----------
    NA

    Revisions
    ---------
    201XXXXX  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    fig4paper
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    # Font settings
    ax.tick_params(labelsize=24)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        try:
            item.set_fontname('Times New Roman')
        except Exception:
            pass

    # Line width for axes
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Color order
    color_order = np.array([
        [0, 0, 255],      # Blue
        [0, 128, 0],      # Green
        [255, 0, 0],      # Red
        [204, 8, 204],    # Purple
        [222, 125, 0],    # Brown
        [0, 51, 153],     # Navy
        [64, 64, 64]      # Gray
    ]) / 255

    ax.set_prop_cycle('color', color_order)

    # Grid
    ax.grid(True)

    # Export
    fig.savefig('fig1.pdf', format='pdf', bbox_inches='tight')


def plot_std(t: np.ndarray, mean_val: np.ndarray, std_val: np.ndarray,
             line_color: str = 'r', region_color: str = 'k',
             ax: Optional[Axes] = None) -> None:
    """
    # Plot with standard deviation

    Parameters
    ----------
    t : np.ndarray
        value along x-axis, Nx1 vector
    mean_val : np.ndarray
        mean value, Nx1 vector
    std_val : np.ndarray
        standard deviation, Nx1 vector
    line_color : str
        line color (default: 'r')
    region_color : str
        region color (default: 'k')
    ax : Axes, optional
        matplotlib axes object (default: current axes)

    Returns
    -------
    None (figure is plotted)

    Notes
    -----
    standard deviationも一緒にplot

    References
    ----------
    NA

    Revisions
    ---------
    20230822  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    fig4paper
    """
    if ax is None:
        ax = plt.gca()

    t = np.asarray(t).flatten()
    mean_val = np.asarray(mean_val).flatten()
    std_val = np.asarray(std_val).flatten()

    # Plot mean line
    ax.plot(t, mean_val, line_color)

    # Fill between for standard deviation
    ax.fill_between(t, mean_val - std_val, mean_val + std_val,
                    color=region_color, alpha=0.1)


def draw_shadow_zones(t: np.ndarray, sunlit_flag: np.ndarray,
                      ax: Optional[Axes] = None, **kwargs) -> None:
    """
    # Draw shadow zones on plot

    Parameters
    ----------
    t : np.ndarray
        time vector
    sunlit_flag : np.ndarray
        sunlit flag (1: sunlit, 0: shadow)
    ax : Axes, optional
        matplotlib axes object (default: current axes)
    **kwargs
        additional arguments passed to axvspan

    Returns
    -------
    None

    Notes
    -----
    sunlitFlagに応じて，xregionでeclipse中であることを示す

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    plot_std
    """
    if ax is None:
        ax = plt.gca()

    t = np.asarray(t).flatten()
    sunlit_flag = np.asarray(sunlit_flag).flatten()

    # Check if sunlitFlag contains any 0
    if np.all(sunlit_flag):
        return

    # Pad with 1 (sunlit) to handle edges
    padded_flag = np.concatenate([[1], sunlit_flag, [1]])
    d = np.diff(padded_flag)

    # -1: transition from 1 to 0 (start of eclipse)
    # 1: transition from 0 to 1 (end of eclipse)
    start_idx = np.where(d == -1)[0]
    end_idx = np.where(d == 1)[0] - 1

    # Default kwargs for shadow zones
    default_kwargs = {'color': 'gray', 'alpha': 0.3}
    default_kwargs.update(kwargs)

    # Apply axvspan for each shadow zone
    for s, e in zip(start_idx, end_idx):
        ax.axvspan(t[s], t[e], **default_kwargs)


def g_figs(n_disp: int = 1) -> None:
    """
    # Grid arrangement of figures on display

    Parameters
    ----------
    n_disp : int
        display number (1 = primary, 2 = secondary, etc.)

    Returns
    -------
    None

    Notes
    -----
    figuresを並べて表示するだけの関数

    References
    ----------
    NA

    Revisions
    ---------
    20250623  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    cls
    """
    if n_disp == 0:
        return

    # Get figure handles
    fig_nums = plt.get_fignums()
    n_figs = len(fig_nums)

    if n_figs == 0:
        warnings.warn('No visible figures.')
        return

    # Try to get screen info (platform dependent)
    try:
        # This is a simplified version - getting actual monitor info
        # requires platform-specific code or additional libraries
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except Exception:
        # Default screen size
        screen_width = 1920
        screen_height = 1080

    # Grid size calculation
    n_row = int(np.ceil(np.sqrt(n_figs)))
    n_col = int(np.ceil(n_figs / n_row))

    gap = 50  # Margin in pixels
    fig_w = (screen_width - gap * (n_col + 1)) / n_col
    fig_h = (screen_height - gap * (n_row + 1)) / n_row

    # Convert to inches
    dpi = 100
    fig_w_inch = fig_w / dpi
    fig_h_inch = fig_h / dpi

    # Arrange figures
    for k, num in enumerate(fig_nums):
        fig = plt.figure(num)

        r = k // n_col  # Row (0-based)
        c = k % n_col   # Column

        # Set figure size
        fig.set_size_inches(fig_w_inch, fig_h_inch)

        # Move figure (requires backend support)
        try:
            mngr = fig.canvas.manager
            left = gap + c * (fig_w + gap)
            top = gap + r * (fig_h + gap)
            mngr.window.wm_geometry(f"+{int(left)}+{int(top)}")
        except Exception:
            pass  # Window positioning not supported


def cls() -> None:
    """
    # Close all figure windows

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    figure windowsを全て閉じる．ただそれだけ．

    References
    ----------
    NA

    Revisions
    ---------
    20230202  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    fig4presen
    """
    plt.close('all')


# %[appendix]{"version":"1.0"}
