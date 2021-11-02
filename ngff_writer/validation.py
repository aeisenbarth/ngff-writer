from typing import Collection, Sequence

from ngff_writer.constants import DIMENSION_AXES


def validate_axes_names(
    axes_names: Sequence[str],
    n_expected_axes: int,
    allowed_axes_names: Collection[str] = DIMENSION_AXES,
):
    if not len(axes_names) == len(set(axes_names)):
        raise ValueError(f"Axes names are not unique: {axes_names}")
    if not set(axes_names) <= set(allowed_axes_names):
        raise ValueError(
            f"Only axes names are allowed from {allowed_axes_names}, not {set(axes_names) - set(allowed_axes_names)}"
        )
    if not len(axes_names) == n_expected_axes:
        raise ValueError(
            f"Axes names must match the number of dimensions: {len(axes_names)} (axes names) != {n_expected_axes} (dimensions)"
        )
