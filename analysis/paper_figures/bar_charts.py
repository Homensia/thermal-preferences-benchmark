"""Grouped bar charts in the layout of the paper's bar figures (Figure_Table3 to 7).

The earlier versions of these figures were drawn with R 4.3.2 / ggplot2 and no
generating code was kept. This module reproduces their chart type and layout
in Matplotlib, read from the PDF/JPEG files of the manuscript:

* ``theme_bw`` look: white panels with a black border, light grey major and
  fainter minor horizontal grid lines, grey strips with bold titles above the
  panels (metric or target) and, in a grid layout, on the right (dataset);
* bars grouped by target on the x-axis (bold tick labels), one fill per model or
  configuration, black bar outlines;
* y-axis 0 to 1 (ticks every 0.1 or 0.2), y-label "Metric Score";
* legend on top, bold title, square keys, three or four columns;
* Okabe-Ito colours as in the PDFs (extracted from their fill operators):
  ``#0072B2 #D55E00 #009E73 #56B4E9 #E69F00 #8FAADC`` (Figure_Table3),
  ``#0072B2 #D55E00 #009E73 #CC79A7 #56B4E9 #E69F00 #332288`` (Figure_Table5),
  ``#4A6572 #344955 #F9AA33 #884EA0 #D55E00`` (Figure_Table6, indices);
* page width 510 pt (7.09 in), Helvetica (Liberation Sans here: same metrics, TrueType).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

FONT_FAMILY = ["Liberation Sans", "DejaVu Sans"]   # Helvetica metrics, TrueType
PAGE_WIDTH_IN = 510.24 / 72
STRIP_FILL = "#EAEAEA"
MAJOR_GRID = "#DCDCDC"
MINOR_GRID = "#F0F0F0"

PHASE1_HAGHIRAD_COLOURS = {
    "Haghirad_NotOpt_NoCW": "#0072B2", "Haghirad_NotOpt_Bal": "#D55E00",
    "Haghirad_NotOpt_BalSub": "#009E73", "Haghirad_Opt_NoCW": "#56B4E9",
    "Haghirad_Opt_Bal": "#E69F00", "Haghirad_Opt_BalSub": "#8FAADC",
}
PHASE1_HIGHCAP_COLOURS = {"HighCap_NoCW": "#0072B2", "HighCap_Bal": "#D55E00", "HighCap_BalSub": "#009E73"}
MODEL_COLOURS = {
    "RF": "#0072B2", "XGBoost": "#D55E00", "SVM": "#009E73", "ANN": "#CC79A7",
    "FTT": "#56B4E9", "FTT+RF": "#E69F00", "FTT+XGBoost": "#332288",
}
INDEX_COLOURS = {"PMV": "#4A6572", "aPMV": "#344955", "ePMV": "#F9AA33", "PTS_SET": "#884EA0",
                 "PTS_PET": "#D55E00"}


def _rc():
    return {"font.family": "sans-serif", "font.sans-serif": FONT_FAMILY, "pdf.fonttype": 42,
            "ps.fonttype": 42, "axes.unicode_minus": False}


def _style_panel(ax, ystep: float, ymax: float, show_y: bool, minor_per_major: int = 5):
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.round(np.arange(0, 1.0 + 1e-9, ystep), 3))
    ax.set_yticks(np.round(np.arange(0, ymax + 1e-9, ystep / minor_per_major), 4), minor=True)
    ax.yaxis.grid(True, which="major", color=MAJOR_GRID, linewidth=0.6, zorder=0)
    ax.yaxis.grid(True, which="minor", color=MINOR_GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(0.7)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.tick_params(axis="both", which="major", length=3, width=0.6, labelsize=9, colors="black")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    if not show_y:
        ax.tick_params(axis="y", labelleft=False, length=0)


def _strip_top(ax, text: str, height_frac: float, fill: str):
    ax.add_patch(Rectangle((0, 1), 1, height_frac, transform=ax.transAxes, clip_on=False,
                           facecolor=fill, edgecolor="black", linewidth=0.7))
    ax.text(0.5, 1 + height_frac / 2, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, fontweight="bold")


STRIP_W_IN = 0.26              # right strip width for one line of text; each further line adds STRIP_LINE_IN
STRIP_LINE_IN = 0.17


def strip_width(labels) -> float:
    lines = max([str(t).count("\n") + 1 for t in labels] or [1])
    return STRIP_W_IN + STRIP_LINE_IN * (lines - 1)


def _strip_right(ax, text: str, width_frac: float, fill: str):
    ax.add_patch(Rectangle((1, 0), width_frac, 1, transform=ax.transAxes, clip_on=False,
                           facecolor=fill, edgecolor="black", linewidth=0.7))
    multi = "\n" in text
    ax.text(1 + width_frac / 2, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=9 if multi else 10.5, fontweight="bold", rotation=270, linespacing=1.15,
            multialignment="center")


def _bars(ax, sub: pd.DataFrame, x_order: Sequence[str], fill_order: Sequence[str],
          colours: Dict[str, str], value_col: str = "value", width: float = 0.84):
    n = len(fill_order)
    w = width / n
    xs = np.arange(len(x_order))
    for j, f in enumerate(fill_order):
        vals = []
        for x in x_order:
            v = sub[(sub["x"] == x) & (sub["fill"] == f)][value_col]
            vals.append(float(v.iloc[0]) if len(v) else np.nan)
        ax.bar(xs - width / 2 + w * (j + 0.5), vals, width=w, color=colours[f], edgecolor="black",
               linewidth=0.45, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_order, fontweight="bold", fontsize=9.5)
    ax.set_xlim(-0.5, len(x_order) - 0.5)


def facet_bars(long: pd.DataFrame, *, panel_order: Sequence[str], x_order: Sequence[str],
               fill_order: Sequence[str], colours: Dict[str, str], legend_title: str,
               row_order: Optional[Sequence[str]] = None, row_labels: Optional[Dict[str, str]] = None,
               ncols: Optional[int] = None, ystep: float = 0.1, ymax: float = 1.0,
               ylabel: str = "Metric Score", height_in: float = 6.4, legend_ncol: int = 3,
               legend_byrow: bool = True, strip_fill: str = STRIP_FILL):
    """Facet grid of grouped bars.

    ``long`` has columns ``panel`` (metric), ``x`` (target), ``fill`` (model or
    configuration), ``value`` and, for a grid, ``row`` (dataset). Grid layout
    (``row_order`` given): rows = datasets with right strips, columns = panels.
    Wrap layout: panels in ``ncols`` columns with top strips only.
    """
    missing = sorted(set(long["fill"]) - set(fill_order))
    if missing:
        raise ValueError(f"fill levels without a colour/order: {missing}")
    with plt.rc_context(_rc()):
        if row_order is not None:
            nr, nc = len(row_order), len(panel_order)
        else:
            nc = ncols or len(panel_order)
            nr = int(np.ceil(len(panel_order) / nc))
        width = PAGE_WIDTH_IN
        strip_w = strip_width([(row_labels or {}).get(r, r) for r in row_order]) if row_order is not None else 0.0
        left, right_pad = 0.62, (0.16 + strip_w if row_order is not None else 0.12)
        top_legend, bottom = 0.95, 0.42
        strip_h, gap_w, gap_h = 0.26, 0.17, (0.17 if row_order is not None else 0.08)
        panel_w = (width - left - right_pad - gap_w * (nc - 1)) / nc
        panel_h = (height_in - top_legend - bottom - strip_h * (1 if row_order is not None else nr)
                   - gap_h * (nr - 1)) / nr
        fig = plt.figure(figsize=(width, height_in))
        axes = []
        for r in range(nr):
            for c in range(nc):
                k = r * nc + c
                if row_order is None and k >= len(panel_order):
                    continue
                strips_above = (1 if row_order is not None else r + 1)
                y0 = height_in - top_legend - strip_h * strips_above - r * (panel_h + gap_h) - panel_h
                x0 = left + c * (panel_w + gap_w)
                ax = fig.add_axes([x0 / width, y0 / height_in, panel_w / width, panel_h / height_in])
                if row_order is not None:
                    panel, rowv = panel_order[c], row_order[r]
                    sub = long[(long["panel"] == panel) & (long["row"] == rowv)]
                    if r == 0:
                        _strip_top(ax, panel, strip_h / panel_h, strip_fill)
                    if c == nc - 1:
                        _strip_right(ax, (row_labels or {}).get(rowv, rowv), strip_w / panel_w, strip_fill)
                else:
                    panel = panel_order[k]
                    sub = long[long["panel"] == panel]
                    _strip_top(ax, panel, strip_h / panel_h, strip_fill)
                _style_panel(ax, ystep, ymax, show_y=(c == 0))
                _bars(ax, sub, x_order, fill_order, colours)
                below = (r < nr - 1) if row_order is not None else (k + nc < len(panel_order))
                if below:                     # ggplot facets: x labels only under the bottom panels
                    ax.tick_params(axis="x", labelbottom=False)
                axes.append(ax)
        fig.text(0.16 / width, (bottom + (height_in - top_legend - bottom) / 2) / height_in, ylabel,
                 rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
        handles = [Patch(facecolor=colours[f], edgecolor="black", linewidth=0.6, label=f) for f in fill_order]
        _top_legend(fig, handles, legend_title, legend_ncol, legend_byrow,
                    y_frac=(height_in - top_legend / 2) / height_in)
    return fig


def _apply_font(fig):
    """Text family is resolved at draw time: set it on every text so it holds outside rc_context."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        t.set_fontfamily(FONT_FAMILY)


def _top_legend(fig, handles, title: str, ncol: int, byrow: bool, y_frac: float):
    """ggplot2-like legend above the panels: bold title on the left of the keys, the pair centred."""
    if byrow:                                   # Matplotlib fills legends column by column
        nrow = int(np.ceil(len(handles) / ncol))
        handles = [handles[r * ncol + c] for c in range(ncol) for r in range(nrow) if r * ncol + c < len(handles)]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, y_frac), ncol=ncol, frameon=False,
                     fontsize=9.5, handlelength=1.4, handleheight=1.4, columnspacing=1.0, handletextpad=0.5,
                     borderaxespad=0.0)
    txt = fig.text(0.0, y_frac, title, ha="left", va="center", fontsize=10.5, fontweight="bold")
    _apply_font(fig)
    renderer = fig.canvas.get_renderer()
    fig_w = fig.get_figwidth() * fig.dpi
    w_title = txt.get_window_extent(renderer).width / fig_w
    w_leg = leg.get_window_extent(renderer).width / fig_w
    gap = 0.01
    if w_title + gap + w_leg <= 0.99:
        x0 = (1.0 - (w_title + gap + w_leg)) / 2
        txt.set_x(x0)
        leg.set_bbox_to_anchor((x0 + w_title + gap, y_frac), transform=fig.transFigure)
        return
    # too wide for the page (Figure_Table3: six long configuration names): the title goes on its own line
    # above the keys, both centred, so the figure stays within the text width
    fig_h = fig.get_figheight() * fig.dpi
    h_title = txt.get_window_extent(renderer).height / fig_h
    h_leg = leg.get_window_extent(renderer).height / fig_h
    vgap = 0.004
    txt.set_ha("center")
    txt.set_position((0.5, y_frac + (h_leg + vgap) / 2))
    leg.set_loc("center")
    leg.set_bbox_to_anchor((0.5, y_frac - (h_title + vgap) / 2), transform=fig.transFigure)


def index_bars(long: pd.DataFrame, *, panel_order: Sequence[str], x_order: Sequence[str],
               colours: Dict[str, str], ylabel: str = "Accuracy Score", height_in: float = 4.53,
               strip_fill: str = STRIP_FILL, tick_labels: Optional[Dict[Tuple[str, str], str]] = None):
    """One panel per target, one bar per index (Figure_Table6 layout, no legend).

    A panel shows only the indices it has (E7b has no TPV rows: the TPV panel then holds PMV and aPMV);
    its bars keep the common width and are centred in the panel. ``tick_labels`` maps (panel, index) to
    the tick label (e.g. the index with its coverage); default: the index name."""
    with plt.rc_context(_rc()):
        width = PAGE_WIDTH_IN
        nc = len(panel_order)
        left, right_pad, top, bottom, strip_h, gap_w = 0.62, 0.12, 0.10, (1.05 if tick_labels else 0.80), 0.26, 0.25
        panel_w = (width - left - right_pad - gap_w * (nc - 1)) / nc
        panel_h = height_in - top - bottom - strip_h
        fig = plt.figure(figsize=(width, height_in))
        for c, panel in enumerate(panel_order):
            ax = fig.add_axes([(left + c * (panel_w + gap_w)) / width, bottom / height_in, panel_w / width,
                               panel_h / height_in])
            sub = long[long["panel"] == panel]
            px = [x for x in x_order if (sub["x"] == x).any()]
            xs = np.arange(len(px)) + (len(x_order) - len(px)) / 2
            vals = [float(sub[sub["x"] == x]["value"].iloc[0]) for x in px]
            ax.bar(xs, vals, width=0.75, color=[colours[x] for x in px], edgecolor="black",
                   linewidth=0.45, zorder=3)
            _strip_top(ax, panel, strip_h / panel_h, strip_fill)
            _style_panel(ax, 0.1, 1.0, show_y=(c == 0))
            ax.set_xticks(xs)
            ax.set_xticklabels([(tick_labels or {}).get((panel, x), x) for x in px], rotation=30, ha="right",
                               rotation_mode="anchor", fontweight="bold", fontsize=10)
            ax.set_xlim(-0.5, len(x_order) - 0.5)
        fig.text(0.16 / width, (bottom + panel_h / 2) / height_in, ylabel, rotation=90, ha="center",
                 va="center", fontsize=11, fontweight="bold")
        _apply_font(fig)
    return fig


def long_from_wide(df: pd.DataFrame, metrics: List[Tuple[str, str]], x_col: str, fill_col: str,
                   row_col: Optional[str] = None) -> pd.DataFrame:
    """Wide metric table -> long frame (panel, x, fill, value[, row])."""
    out = []
    for col, label in metrics:
        part = pd.DataFrame({"panel": label, "x": df[x_col].values, "fill": df[fill_col].values,
                             "value": df[col].astype(float).values})
        if row_col:
            part["row"] = df[row_col].values
        out.append(part)
    return pd.concat(out, ignore_index=True)
