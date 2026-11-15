"""Wording guard for every label, title, legend, caption and file written by the package.

Banned (the near-neighbour agreement is not a limit on accuracy): "ceiling", "upper bound",
"demonstrate" (any inflection), "informational rather than algorithmic".
Terminology: PMV, SET and PET are heat-balance models, aPMV and ePMV adaptive
heat-balance models; "empirical" is never used for them. The French cohorts are
named Moujalled and Hostein (never by their former names).
"""
from __future__ import annotations

import re

BANNED_PATTERNS = (
    (re.compile(r"ceiling", re.I), "ceiling"),
    (re.compile(r"upper[\s_-]*bound", re.I), "upper bound"),
    (re.compile(r"demonstrat", re.I), "demonstrate"),
    (re.compile(r"informational\s+rather\s+than\s+algorithmic", re.I), "informational rather than algorithmic"),
    (re.compile(r"empiric", re.I), "empirical (heat-balance terminology)"),
    (re.compile(r"cerema|mathilde", re.I), "former cohort name (use Moujalled / Hostein)"),
)


class BannedWordingError(ValueError):
    """A label, caption or file name contains banned wording."""


def banned_in(text: str) -> list:
    return [name for pattern, name in BANNED_PATTERNS if pattern.search(text or "")]


def check_text(text: str, where: str = "") -> None:
    found = banned_in(text)
    if found:
        raise BannedWordingError(f"{where}: banned wording {found} in {text[:200]!r}")


def figure_texts(fig) -> list:
    """Every text drawn by a Matplotlib figure (titles, labels, ticks, legends, annotations)."""
    from matplotlib.text import Text
    fig.canvas.draw()                      # tick labels are populated at draw time
    return [t.get_text() for t in fig.findobj(Text)]


def check_figure(fig, where: str = "") -> None:
    for text in figure_texts(fig):
        check_text(text, where)
