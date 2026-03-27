
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm

def histplot(
    x: np.ndarray,
    y: np.ndarray,
    output_path: Union[str, Path],
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None,
    cmap: str = 'viridis_r',
    figsize: Tuple[float, float] = (8, 8),
    dpi: int = 600,
    watermark_text: Optional[str] = None,
    info_texts: Optional[list] = None,
    save_pdf: bool = True,
    save_png: bool = True) -> plt.Figure:
    """
    Create a 2d histplot visualization.
    
    Parameters
    ----------
    data : np.ndarray
        2D array to visualize (e.g., confusion matrix)
    output_path : str or Path
        Base output path (without extension). Will save .png and/or .pdf
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    title : str, optional
        Title for the plot
    cmap : str, default='viridis_r'
        Colormap name
    figsize : tuple, default=(8, 8)
        Figure size in inches (width, height)
    dpi : int, default=600
        DPI for saved PNG image
    watermark_text : str, optional
        Watermark text to add (e.g., "2025 (own work)")
    info_texts : list of dict, optional
        List of text annotations. Each dict should have 'text', 'px', 'py' keys
        Example: [{'text': 'Experiment 1', 'px': 0.05, 'py': 0.87}]
    save_pdf : bool, default=True
        Whether to save PDF version
    save_png : bool, default=True
        Whether to save PNG version
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    """
    
    if x.shape != y.shape:
        raise ValueError(f"Arrays must have the same shape. Got {x.shape} and {y.shape}")
    
    flat1 = x.flatten()
    flat2 = y.flatten()
    
    valid_mask = ~(np.isnan(flat1) | np.isnan(flat2))
    flat1 = flat1[valid_mask]
    flat2 = flat2[valid_mask]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if watermark_text is not None:
        ax.text(0.05, 0.92, watermark_text, 
                transform=ax.transAxes,
                fontsize=10, alpha=0.6,
                verticalalignment='top')
    
    if info_texts is not None:
        for info in info_texts:
            ax.text(info['px'], info['py'], info['text'],
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top')
    
    cbar_ax = inset_axes(ax,
                        width="3%",       
                        height="80%",
                        loc='center right',
                        bbox_to_anchor=(0.15, 0., 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    
    histplot = sns.histplot(x=flat1,
                            y=flat2,
                            bins=100,
                            cbar=True,
                            cmap=cmap,
                            stat="count",
                            ax=ax,
                            cbar_ax=cbar_ax,
                            norm=LogNorm(), 
                            vmin=None, 
                            vmax=None)

    histplot.margins(x=0.05, y=0.05)
    
    cbar = histplot.figure.colorbar(histplot.collections[0], cax=cbar_ax)
    cbar.set_label("Count")
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    
    ax.set_box_aspect(1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title is not None:
        ax.set_title(title)
    
    plt.subplots_adjust(left=0.175, right=0.75, top=0.95, bottom=0.05)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_png:
        fig.savefig(f"{output_path}.png", dpi=dpi)
    
    if save_pdf:
        fig.savefig(f"{output_path}.pdf")
    
    plt.close(fig)
    
    return fig