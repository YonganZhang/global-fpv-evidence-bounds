"""Small, repository-local publication figure helpers.

The active v117 figure builder only needs :func:`normalize_fonts`.  Keeping
the helper beside the plotting code removes a hidden dependency on a local
Codex installation and makes the public release self-contained.
"""

from __future__ import annotations

import re


FONT_SIZES = {
    "panel_label": 12,
    "title": 11,
    "axis_label": 10,
    "tick": 9,
    "legend": 8.5,
    "annotation": 9,
    "small": 7.5,
    "colorbar": 8,
}


def _is_panel_label(text: str) -> bool:
    return bool(re.match(r"^\(?[a-zA-Z]\)?$", text.strip()))


def _is_country_code(text: str) -> bool:
    value = text.strip()
    return bool(re.match(r"^[A-Z]{2,3}$", value)) and "\n" not in value


def _scale_for_axis(axis, base_size: float) -> float:
    """Scale text gently for small subplots without shrinking panel labels."""
    try:
        relative_width = axis.get_position().width
        return base_size * (0.8 + 0.2 * relative_width)
    except (AttributeError, TypeError, ValueError):
        return base_size


def normalize_fonts(
    figure,
    sizes: dict[str, float] | None = None,
    scale_by_subplot: bool = False,
) -> None:
    """Normalize all existing text objects before a figure is exported.

    The optional subplot scaling matches the public builder's call signature
    and keeps the repository-local helper compatible with the canonical
    publication standard without depending on a user-specific Codex path.
    """
    font_sizes = {**FONT_SIZES, **(sizes or {})}
    if getattr(figure, "_suptitle", None) is not None:
        figure._suptitle.set_fontsize(font_sizes["title"])

    for axis in figure.get_axes():
        def size(key: str) -> float:
            base = font_sizes[key]
            return _scale_for_axis(axis, base) if scale_by_subplot else base

        if axis.title.get_text():
            axis.title.set_fontsize(size("title"))
        axis.xaxis.label.set_fontsize(size("axis_label"))
        axis.yaxis.label.set_fontsize(size("axis_label"))
        axis.tick_params(axis="both", which="major", labelsize=size("tick"))

        legend = axis.get_legend()
        if legend:
            for item in legend.get_texts():
                item.set_fontsize(size("legend"))
            if legend.get_title() and legend.get_title().get_text():
                legend.get_title().set_fontsize(size("legend"))

        for container in axis.containers:
            for item in container.get_children():
                if hasattr(item, "set_fontsize"):
                    item.set_fontsize(size("annotation"))

        for item in axis.texts:
            value = item.get_text().strip()
            if not value:
                continue
            if _is_panel_label(value):
                item.set_fontsize(font_sizes["panel_label"])
                item.set_fontweight("bold")
            elif _is_country_code(value):
                item.set_fontsize(size("small"))
            else:
                item.set_fontsize(size("annotation"))

        if getattr(axis, "_colorbar_info", None) is not None or axis.get_label() == "<colorbar>":
            axis.tick_params(labelsize=font_sizes["tick"])
            axis.xaxis.label.set_fontsize(font_sizes["colorbar"])
            axis.yaxis.label.set_fontsize(font_sizes["colorbar"])

    for item in figure.texts:
        value = item.get_text().strip()
        if not value:
            continue
        if _is_panel_label(value) or (len(value) <= 2 and value.isalpha()):
            item.set_fontsize(font_sizes["panel_label"])
            item.set_fontweight("bold")
        else:
            item.set_fontsize(font_sizes["annotation"])
