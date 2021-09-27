import math
from typing import Callable, Sequence, Union

import dask_image.ndfilters
import numpy as np
import skimage.filters
import skimage.transform
import zarr
from dask import array as da

from ngff_writer.constants import DIMENSION_AXES, SPATIAL_DIMENSIONS
from ngff_writer.dask_utils import downscale_nearest
from ngff_writer.typing import DimensionAxisType


def apply_over_axes(
    func: Callable[[Union[np.ndarray, da.Array]], Union[np.ndarray, da.Array]],
    array: Union[np.ndarray, da.Array],
    axes: Sequence[int],
) -> Union[np.ndarray, da.Array]:
    """
    Apply a function over sliced subarrays of the specified axes. Iterate over all such subarrays
    along the other axes.

    Args:
        func: Function to apply to the sliced subarray.
        array: Input array.
        axes: Axes indices of the input array which become the only axes of the subarray

    Returns:
        Modified input array.
    """
    selected_axes_shape = tuple(array.shape[i] for i in axes)
    target_axes_indices = tuple(range(array.ndim - len(axes), array.ndim))
    # Reshape the array so that the specified axes for the subarrays are grouped and all remaining axes are collapsed into one.
    array_reordered = np.moveaxis(array, axes, target_axes_indices)
    reordered_shape = array_reordered.shape
    array_reshaped = array_reordered.reshape((-1,) + selected_axes_shape)
    # Apply the function over all subarrays by iterating over the remaining axes.
    # Be aware: concatenate squeezes 1-dimensions
    array_processed = np.stack([func(subarray).astype(array.dtype) for subarray in array_reshaped])
    reordered_processed_shape = reordered_shape[: -len(axes)] + array_processed.shape[1:]
    # Restore the array
    return np.moveaxis(
        array_processed.reshape(reordered_processed_shape), target_axes_indices, axes
    )


def ngff_spatially_rescale(
    image: Union[np.ndarray, da.Array],
    scale: float,
    axes_names: Sequence[DimensionAxisType] = DIMENSION_AXES,
    is_label: bool = False,
) -> Union[np.ndarray, da.Array]:
    """
    Rescales an NGFF image taking into account only spatial dimensions.

    Args:
        image: Input image.
        scale: Upscale factor.
        axes_names: List of NGFF axis names matching the input image's axes
        is_label: Whether this is a label image and

    Returns:
        NGFF image with spatial dimensions rescaled and other dimensions unchanged.
    """
    assert scale > 0
    assert len(axes_names) == len(set(axes_names))
    assert set(axes_names) <= set(DIMENSION_AXES)
    assert image.ndim == len(axes_names)
    if isinstance(image, da.Array):
        # _resize = dask_resize
        factors = tuple(
            min(dim, round(1 / scale)) if name in SPATIAL_DIMENSIONS else 1
            for dim, name in zip(image.shape, axes_names)
        )
        _resize = lambda image, out_shape, **kwargs: downscale_nearest(image, factors)
        _smooth = dask_image.ndfilters.gaussian_filter
    else:
        _resize = skimage.transform.resize
        _smooth = skimage.filters.gaussian  # scipy.ndimage.gaussian_filter
    out_shape = tuple(
        math.ceil(dim * float(scale)) if name in SPATIAL_DIMENSIONS else dim
        for dim, name in zip(image.shape, axes_names)
    )
    if is_label:
        output = _resize(
            image, out_shape, order=0, mode="reflect", anti_aliasing=False, preserve_range=True
        ).astype(image.dtype)
        if isinstance(image, np.ndarray):
            assert set(np.unique(output)) <= set(np.unique(image))
        return output
    else:
        spatial_axes = [i for i, name in enumerate(axes_names) if name in SPATIAL_DIMENSIONS]
        if scale < 1:
            sigma = 2 / 6.0 / scale

            def spatial_smooth(spatial_array):
                squeezed_array = spatial_array.squeeze()
                sigmas = tuple(min(sigma, d / 4.0) for d in squeezed_array.shape)
                return _smooth(squeezed_array, sigmas, mode="reflect").reshape(spatial_array.shape)

            image = apply_over_axes(spatial_smooth, image, spatial_axes)
        output = _resize(
            image.astype(float), out_shape, order=1, mode="reflect", anti_aliasing=False
        )
        if scale > 1:
            sigma = 2 * scale / 6.0
            output = apply_over_axes(
                lambda subarray: _smooth(subarray, sigma, mode="reflect"), output, spatial_axes
            )
        return output.astype(image.dtype)


def select_dimensions(
    array: Union[np.ndarray, da.Array, zarr.Array],
    selected_dimension_axes: Sequence[str],
    all_dimension_axes: Sequence[str],
) -> Union[np.ndarray, da.Array]:
    """
    Slices an array so that only selected dimensions in specified order are returned.

    Args:
        array: A Numpy or Dask array
        selected_dimension_axes: A list of dimension names to include in the returned array. If
            dimensions are dropped they must have length 1, otherwise an error is raised.
        all_dimension_axes: A list of all the original array's dimension names

    Returns:
        An array with only selected dimensions in specified order
    """
    assert len(set(selected_dimension_axes)) == len(selected_dimension_axes)
    assert len(set(all_dimension_axes)) == len(all_dimension_axes)
    assert set(selected_dimension_axes) <= set(all_dimension_axes)
    slices = [slice(None)] * 5
    sliced_axes = []
    for d, dim_name in enumerate(all_dimension_axes):
        if dim_name not in selected_dimension_axes:
            assert array.shape[d] == 1
            slices[d] = 0
        else:
            sliced_axes.append(dim_name)
    # Slicing and transposition convert Zarr arrays to Numpy arrays!
    sliced_array = array[tuple(slices)]
    if tuple(sliced_axes) == tuple(selected_dimension_axes):
        return sliced_array
    else:
        axes = [sliced_axes.index(dim_name) for dim_name in selected_dimension_axes]
        # If selected axes order differs from original axis order, transposition is required.
        return sliced_array.transpose(axes)


def to_tczyx(
    array: Union[np.ndarray, da.Array], axes_names: Sequence[DimensionAxisType]
) -> Union[np.ndarray, da.Array]:
    """
    Converts a data array matrix of a subset of NGFF dimensions in different order to an ordered
    5-dimensional TCZYX array.

    Args:
        array: The data array.
        axes_names: The axes in the order used by the array, for example ("x", "y"). Axes names
            must be of "t", "c", "z", "y", "x".

    Returns:
        A 5-dimensional array
    """
    assert len(axes_names) == len(set(axes_names))
    assert set(axes_names) <= set(DIMENSION_AXES)
    assert array.ndim == len(axes_names)
    ordered_axes_names = [axis for axis in DIMENSION_AXES if axis in axes_names]
    axes_indices = [axes_names.index(axis) for axis in ordered_axes_names]
    shape = tuple(
        array.shape[axes_names.index(axis)] if axis in axes_names else 1 for axis in DIMENSION_AXES
    )
    return array.transpose(axes_indices).reshape(shape)
