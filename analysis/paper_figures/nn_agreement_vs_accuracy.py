"""Test accuracy of the Phase 2 models beside the near-neighbour vote agreement.

Restored from the figure script of the pre-revision code (``analysis/performance_ceiling.py``).
The three-panel layout, the colours, the bar annotations, the two vertical
reference lines and the shared legend are kept (``_panel`` and the figure
assembly are the pre-revision code). Minimal adaptation, with reasons:

* file name and labels renamed (``nn_agreement_vs_accuracy``): the
  near-neighbour agreement is a descriptive statistic of the votes, not a
  limit on the accuracy of a model (banned wording, ``wording.py``);
* inputs: E1 seed-mean accuracies (``aggregate.py``) and, when E10 is complete, the
  FTT+RF / FTT+XGB heads, instead of the pre-revision ``*_test_results.json``;
  whiskers show the range over seeds 42-44 (the pre-revision figure showed a Wilson
  interval of one run; the revision reports seed means with their range);
* red dashed line: ``1 - D_1`` of E9b, the paper's NN analysis (pre-revision
  primary 1-NN: Euclidean distance on the 12 standardised variables F12, the
  full cohort, each vote against the other votes; category-normalised data), replacing
  the pre-revision constants 0.43/0.56/0.59 by the rerun values;
  in a ``--preview`` without E9b the line is left out;
* grey dotted line: chance agreement ``sum_c p_c**2`` (two votes drawn at
  random from the class distribution), with ``p_c`` the class shares of the
  cohort votes (the queries of ``D_1``), replacing the pre-revision "random
  baseline" ``sum_k p_k^2`` of the RF test priors;
* drawn at its print size: the full text width of the 5p
  layout (510 pt), every text at 7 pt or more at 100 % scale; the population and
  definitions of the reference lines moved to the generated caption
  (``captions/nn_agreement_vs_accuracy.tex``; an optional ``note`` is still drawn
  under the legend when given);
* models in one fixed order in every panel (the table order: RF, XGBoost, SVM,
  ANN, FTT, FTT+RF, FTT+XGBoost) instead of sorted by accuracy, so that the
  panels share their y labels and the figure does not present a ranking; one
  x scale shared by the three panels;
* the value label of each bar sits right of its whisker; the whisker is the
  range over the model seeds (legend entry; a model with one seed has none);
* optional (``cases``, CLI ``--nn_cases_in_figure``): the E9b Case B group agreement and Case C pair agreement (pre-revision
  definitions) as two more vertical lines, each with its own chance
  reference on the same groups / pairs (open circle, and printed in the label)
  and its number of groups / pairs; the whole-cohort chance line does not
  apply to them.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .bar_charts import FONT_FAMILY, PAGE_WIDTH_IN  # noqa: E402
from .metrics import TARGET_LABEL, TARGETS  # noqa: E402

STANDARD_COLOUR = "#4C72B0"
HYBRID_COLOUR = "#8DA0CB"
NN_COLOUR = "#C44E52"
CHANCE_COLOUR = "#7F7F7F"
CASE_B_COLOUR = "#8172B2"
CASE_C_COLOUR = "#937860"


MIN_FONT = 7.0            # pt at print size (Elsevier artwork guidance: 6-7 pt or more)
FS_SMALL, FS_TEXT, FS_TITLE = 7.5, 8.0, 9.0


def _rc():
    return {"font.family": "sans-serif", "font.sans-serif": FONT_FAMILY, "pdf.fonttype": 42, "ps.fonttype": 42}


def _panel(ax, df_t: pd.DataFrame, order, nn_agreement: float, chance_agreement: float, title: str,
           xmax: float, cases=None):
    present = set(df_t["model_display"])
    d = df_t.set_index("model_display").reindex([m for m in order if m in present]).reset_index()
    y_pos = np.arange(len(d))[::-1]                  # first model of ``order`` on top
    colors = [STANDARD_COLOUR if k == "standard" else HYBRID_COLOUR for k in d["kind"]]
    lo_err = (d["accuracy"] - d["acc_low"]).clip(lower=0).fillna(0)
    hi_err = (d["acc_high"] - d["accuracy"]).clip(lower=0).fillna(0)
    ax.barh(y_pos, d["accuracy"].to_numpy(dtype=float),
            xerr=np.vstack([lo_err.to_numpy(dtype=float), hi_err.to_numpy(dtype=float)]), color=colors, edgecolor="black", linewidth=0.4,
            ecolor="#333333", capsize=1.8, error_kw={"elinewidth": 0.8, "capthick": 0.8}, height=0.72)
    for yi, v, hi in zip(y_pos, d["accuracy"], d["acc_high"].fillna(d["accuracy"])):
        ax.text(max(v, hi) + 0.012, yi, f"{v*100:.0f}%", va="center", ha="left", fontsize=FS_SMALL)
    top = len(d) - 0.35
    if pd.notna(nn_agreement):                     # NaN only in a --preview without E9b
        ax.axvline(nn_agreement, color=NN_COLOUR, linestyle="--", linewidth=1.3)
        ax.text(nn_agreement, top, f"{nn_agreement*100:.0f}%", color=NN_COLOUR, fontsize=FS_SMALL,
                ha="center", va="bottom")
    ax.axvline(chance_agreement, color=CHANCE_COLOUR, linestyle=":", linewidth=1.3)
    ax.text(chance_agreement, top, f"{chance_agreement*100:.0f}%", color=CHANCE_COLOUR, fontsize=FS_SMALL,
            ha="center", va="bottom")
    bottom = -0.6
    if cases is not None:
        # each value is printed with its own chance reference (same groups / same pairs: the whole-cohort
        # chance line does not apply to them) and its count (Case B rests on few groups, so the
        # percentage alone overstates it); the chance value is also marked by an open circle
        for value, chance, count, unit, colour, style, y in (
                (cases["case_b_cluster_agreement"], cases["case_b_chance_cluster_agreement"],
                 cases["case_b_clusters"], "groups", CASE_B_COLOUR, "-.", -0.8),
                (cases["case_c_agreement"], cases["case_c_chance_agreement"], cases["case_c_pairs"], "pairs",
                 CASE_C_COLOUR, (0, (5, 2, 1, 2)), -1.6)):
            if pd.notna(value):
                if pd.isna(chance):
                    raise ValueError("a Case B / Case C value without its chance reference")
                ax.axvline(float(value), color=colour, linestyle=style, linewidth=1.2)
                ax.plot([float(chance)], [y + 0.15], marker="o", markersize=4, markerfacecolor="white",
                        markeredgecolor=colour, markeredgewidth=1.0, linestyle="none", zorder=4)
                ax.text(float(value), y, (f"{float(value)*100:.0f}% (chance {float(chance)*100:.0f}%; "
                                          f"{int(count):,} {unit})").replace(",", " "),
                        color=colour, fontsize=MIN_FONT, ha="center", va="top",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        bottom = -2.4                     # room for the two value labels inside the axes, below the bars
    ax.set_ylim(bottom, len(d) - 0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["model_display"], fontsize=FS_TEXT)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.round(np.arange(0, xmax + 1e-9, 0.2), 1))
    ax.tick_params(axis="both", labelsize=FS_TEXT, length=2.5, width=0.6)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=FS_TITLE, loc="left", fontweight="bold", pad=11)


def x_extent(acc: pd.DataFrame, references: pd.DataFrame, cases: pd.DataFrame = None) -> float:
    """One x scale for the three panels: the largest drawn value plus room for its label, to the next 0.1."""
    vals = list(acc["accuracy"]) + list(acc["acc_high"].dropna()) + list(references["nn_agreement"].dropna()) + \
        list(references["chance_agreement"].dropna())
    if cases is not None:
        for c in ("case_b_cluster_agreement", "case_b_chance_cluster_agreement", "case_c_agreement",
                  "case_c_chance_agreement"):
            vals += list(pd.to_numeric(cases[c], errors="coerce").dropna())
    return float(np.ceil((max(vals) + 0.09) * 10) / 10)


def make_figure(acc: pd.DataFrame, references: pd.DataFrame, k: int = 1, note: str = "",
                cases: pd.DataFrame = None, order=None):
    """``acc``: target, model_display, kind (standard/hybrid), accuracy, acc_low, acc_high (the range over
    seeds; equal to the accuracy, or NaN, for a model with one seed). ``references``: target, nn_agreement
    (1 - D_k), chance_agreement. ``order``: the model order, top to bottom, in every panel (default: the
    order of first appearance in ``acc``). ``note`` (optional): text under the legend; the paper figure
    has none, its caption carries the definitions. ``cases`` (optional, indexed by target):
    case_b_cluster_agreement, case_b_chance_cluster_agreement, case_b_clusters, case_c_agreement,
    case_c_chance_agreement, case_c_pairs (E9b, full cohort); each value is printed with its own chance
    reference (open circle and label) and its number of groups (Case B) or pairs (Case C).

    Drawn at its print size: ``PAGE_WIDTH_IN`` wide (510 pt, include at ``width=\\linewidth`` of the
    5p layout, i.e. at 100 %), no text below ``MIN_FONT``."""
    ref = references.set_index("target")
    order = list(dict.fromkeys(acc["model_display"])) if order is None else list(order)
    xmax = x_extent(acc, references, cases)
    height = 3.25 + (0.75 if cases is not None else 0.0)
    with plt.rc_context(_rc()):
        # canvas 0.04 in narrower than the text width: the saved page adds a 0.02 in pad on each side
        fig, axes = plt.subplots(1, 3, figsize=(PAGE_WIDTH_IN - 0.04, height), sharey=False, layout="constrained")
        first = True
        for ax, target in zip(axes, TARGETS):
            df_t = acc[acc["target"] == target]
            if df_t.empty:
                ax.set_visible(False)
                continue
            _panel(ax, df_t, order, nn_agreement=float(ref.loc[target, "nn_agreement"]),
                   chance_agreement=float(ref.loc[target, "chance_agreement"]), title=TARGET_LABEL[target],
                   xmax=xmax, cases=None if cases is None else cases.loc[target])
            if not first:                               # same model order: y labels on the first panel only
                ax.tick_params(axis="y", labelleft=False)
            first = False
        visible = [ax for ax in axes if ax.get_visible()]
        visible[len(visible) // 2].set_xlabel("Test accuracy (bars) and vote agreement (lines)", fontsize=FS_TEXT)
        handles = [
            Patch(facecolor=STANDARD_COLOUR, edgecolor="black", label="Standard model"),
            Patch(facecolor=HYBRID_COLOUR, edgecolor="black", label="Hybrid model (FTT embeddings)"),
            Line2D([0], [0], color="#333333", marker="|", markersize=6, linewidth=0.8,
                   label="Range over model seeds"),
            Line2D([0], [0], color=NN_COLOUR, linestyle="--", linewidth=1.3,
                   label=f"{k}-NN vote agreement (1 − D$_{{{k}}}$)"),
            Line2D([0], [0], color=CHANCE_COLOUR, linestyle=":", linewidth=1.3,
                   label="Chance agreement (two random votes)"),
        ]
        if cases is not None:
            handles += [Line2D([0], [0], color=CASE_B_COLOUR, linestyle="-.", linewidth=1.2,
                               label="Case B: identical feature vectors, group agreement"),
                        Line2D([0], [0], color=CASE_C_COLOUR, linestyle=(0, (5, 2, 1, 2)), linewidth=1.2,
                               label="Case C: pairs within tight margins, agreement"),
                        Line2D([0], [0], color="#555555", marker="o", markersize=4, markerfacecolor="white",
                               linestyle="none", label="Case B / C chance (same groups / pairs)")]
        if ref["nn_agreement"].isna().all():
            handles = [h for h in handles if not h.get_label().startswith(f"{k}-NN")]   # preview without E9b
        if not (acc["kind"] == "hybrid").any():
            handles = [h for h in handles if not h.get_label().startswith("Hybrid")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3 if cases is None else 2,
                   fontsize=FS_TEXT, frameon=False, handlelength=2.2, columnspacing=1.4)
        if note:
            fig.text(0.5, 0.0, note, ha="center", va="top", fontsize=FS_SMALL, color="#333333")
        for t in fig.findobj(matplotlib.text.Text):
            t.set_fontfamily(FONT_FAMILY)
    return fig
