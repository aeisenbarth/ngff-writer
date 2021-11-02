import re
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from dask import array as da
from dask.delayed import Delayed

from ngff_writer.constants import (
    DIMENSION_AXES,
    DIMENSION_SEPARATOR,
    NGFF_SPEC_VERSION,
    SPATIAL_DIMENSIONS,
    ZARR_DISALLOWED_CHARS_REGEX,
)
from ngff_writer.array_utils import ngff_spatially_rescale
from ngff_writer.typing import DimensionAxisType, DimensionSeparatorType


def add_image(
    group: zarr.Group,
    array: Union[np.ndarray, da.Array] = None,
    dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
    chunks: Optional[Sequence[int]] = None,
    channel_names: Optional[Sequence[str]] = None,
    # TODO: channel_metadata: Maybe better dict with channel names as keys?
    channel_metadata: Optional[Sequence[dict]] = None,
    n_scales: Optional[int] = None,
    transformation: Optional[dict] = None,
    overwrite: bool = False,
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
) -> "spacem_lib.io.writer.NgffImage":
    add_omero_metadata(group=group, channel_names=channel_names, channel_metadata=channel_metadata)

    if array is not None:
        set_image_array(
            array=array,
            image_group=group,
            dimension_axes=dimension_axes,
            chunks=chunks,
            n_scales=n_scales,
            transformation=transformation,
            channel_names=channel_names,
            channel_metadata=channel_metadata,
            overwrite=overwrite,
            dimension_separator=dimension_separator,
        )

    from ngff_writer.datastructures import NgffImage

    return NgffImage(group=group, overwrite=overwrite, dimension_separator=dimension_separator)


def set_image_array(
    array: Union[np.ndarray, da.Array],
    image_group: Optional[zarr.Group],
    dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
    chunks: Optional[Sequence[int]] = None,
    n_scales: Optional[int] = None,
    transformation: Optional[dict] = None,
    channel_names: Optional[Sequence[str]] = None,
    channel_metadata: Optional[Sequence[dict]] = None,
    overwrite: bool = False,
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
):
    # Assertions
    assert array.ndim <= len(DIMENSION_AXES)
    if dimension_axes is None:
        dimension_axes = DIMENSION_AXES[-array.ndim :]
    assert len(dimension_axes) == array.ndim
    assert set(dimension_axes) <= set(DIMENSION_AXES)
    assert chunks is None or len(chunks) == array.ndim
    assert n_scales is None or 1 <= 2 ** n_scales <= max(array.shape)
    if channel_names is None:
        assert_omero_channel_names(image_group, array.shape[dimension_axes.index("c")])

    # Defaults
    if chunks is None:
        if image_group.attrs.get("multiscales", [{}])[0].get("datasets"):
            # When overwriting array on existing image, keep same chunking
            max_scale_image = image_group[get_first_array_key(image_group)]
            chunks = max_scale_image.chunks
        else:
            chunks = get_chunks(array.shape, dimension_axes)
    if n_scales is None:
        if image_group.attrs.get("multiscales", [{}])[0].get("datasets"):
            # When overwriting array on existing image, keep same number of scales
            n_scales = len(image_group.attrs["multiscales"][0]["datasets"])
        else:
            n_scales = (
                max(1, int(np.ceil(np.log2(max(array.shape))) - np.floor(np.log2(max(chunks)))))
                if n_scales is None
                else n_scales
            )
    if transformation is None:
        transformation = image_group.attrs.get("multiscales", [{}])[0].get("_transformation")

    # Add data arrays, create pyramids
    add_multiscales(
        group=image_group,
        array=array,
        dimension_axes=dimension_axes,
        chunks=chunks,
        n_scales=n_scales,
        transformation=transformation,
        dimension_separator=dimension_separator,
        overwrite=overwrite,
    )

    # Metadata
    if channel_names is not None or channel_metadata is not None:
        add_omero_metadata(
            group=image_group, channel_names=channel_names, channel_metadata=channel_metadata
        )


def add_label(
    group: zarr.Group,
    array: Union[np.ndarray, da.Array] = None,
    name: Optional[str] = None,
    dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
    chunks: Optional[Sequence[int]] = None,
    n_scales: Optional[int] = None,
    transformation: Optional[dict] = None,
    colors: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    properties: Optional[Sequence[dict]] = None,
    overwrite: bool = False,
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
) -> "spacem_lib.io.writer.NgffLabel":
    if name is not None and re.match(ZARR_DISALLOWED_CHARS_REGEX, name):
        raise ValueError("Label image name contains disallowed characters.")

    labels_group: zarr.Group = group.require_group("labels")
    name = str(len(list(labels_group.group_keys()))) if name is None else name
    label_group: zarr.Group = labels_group.require_group(name)

    if array is not None:
        set_label_array(
            array=array,
            label_group=label_group,
            labels_group=labels_group,
            image_group=group,
            dimension_axes=dimension_axes,
            chunks=chunks,
            n_scales=n_scales,
            transformation=transformation,
            overwrite=overwrite,
            dimension_separator=dimension_separator,
        )

    # Metadata
    add_label_metadata(group=label_group, colors=colors, properties=properties)

    from ngff_writer.datastructures import NgffLabel

    return NgffLabel(
        group=label_group,
        name=name,
        labels_group=labels_group,
        image_group=group,
        overwrite=overwrite,
        dimension_separator=dimension_separator,
    )


def set_label_array(
    array: Union[np.ndarray, da.Array],
    label_group: zarr.Group,
    labels_group: Optional[zarr.Group] = None,
    image_group: Optional[zarr.Group] = None,
    dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
    chunks: Optional[Sequence[int]] = None,
    n_scales: Optional[int] = None,
    transformation: Optional[dict] = None,
    overwrite: bool = False,
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
):
    # Assertions
    assert array.ndim <= len(DIMENSION_AXES)
    if dimension_axes is None:
        dimension_axes = DIMENSION_AXES[-array.ndim :]
    assert len(dimension_axes) == array.ndim
    assert set(dimension_axes) <= set(DIMENSION_AXES)
    assert chunks is None or len(chunks) == array.ndim
    assert n_scales is None or 1 <= 2 ** n_scales <= max(array.shape)
    assert None not in (chunks, n_scales, transformation) or image_group is not None
    assert (
        image_group is None
        or "multiscales" in image_group.attrs
        and len(image_group.attrs["multiscales"]) > 0
    )

    # Defaults
    if chunks is None:
        if label_group.attrs.get("multiscales", [{}])[0].get("datasets"):
            # When overwriting array on existing label, keep same chunking
            max_scale_image = label_group[get_first_array_key(label_group)]
        else:
            # Otherwise use same chunking as image
            max_scale_image = image_group[get_first_array_key(image_group)]
        assert array.ndim == max_scale_image.ndim
        chunks = max_scale_image.chunks
    if n_scales is None:
        if label_group.attrs.get("multiscales", [{}])[0].get("datasets"):
            # When overwriting array on existing label, keep same number of scales
            n_scales = len(label_group.attrs["multiscales"][0]["datasets"])
        else:
            # Otherwise use same number of scales as image
            n_scales = len(image_group.attrs["multiscales"][0]["datasets"])
            n_scales = (
                max(1, int(np.ceil(np.log2(max(array.shape))) - np.floor(np.log2(max(chunks)))))
                if n_scales is None
                else n_scales
            )
    if transformation is None:
        if label_group.attrs.get("multiscales", [{}])[0].get("_transformation"):
            # When overwriting array on existing label, keep same transformation
            transformation = label_group.attrs["multiscales"][0].get("_transformation")
        elif len(image_group.attrs.get("multiscales", [])) >= 1:
            # Otherwise use same transformation as image
            transformation = image_group.attrs["multiscales"][0].get("_transformation")

    # Add data arrays, create pyramids
    add_multiscales(
        group=label_group,
        array=array,
        dimension_axes=dimension_axes,
        chunks=chunks,
        n_scales=n_scales,
        transformation=transformation,
        dimension_separator=dimension_separator,
        overwrite=overwrite,
        is_label=True,
    )
    # Assert that multiscales have same values (no interpolation or smooth sampling).
    assert_multiscale_same_values(
        label_group, [d["path"] for d in label_group.attrs["multiscales"][-1]["datasets"]]
    )

    # Metadata
    if labels_group is None:
        labels_group = zarr.Group(store=label_group.store, path=str(Path(label_group.path).parent))
    add_labels_metadata(group=labels_group, data_path=label_group.basename)


def add_multiscales(
    group: zarr.Group,
    array: Union[np.ndarray, da.Array],
    dimension_axes: Sequence[DimensionAxisType],
    chunks: Sequence[int],
    n_scales: int,
    transformation: Optional[dict],
    dimension_separator: DimensionSeparatorType,
    overwrite: bool,
    is_label: bool = False,
):
    """
    Adds an array as scale pyramid to a Zarr group.

    Args:
        group: The Zarr group
        array: A Numpy or Dask array to write to Zarr
        dimension_axes: The sequence of used dimension axes.
        chunks: Chunk shape. If not provided, will be guessed from `shape` and `dtype`.
        n_scales: The number of scales (including unscaled) arrays in the pyramid
        transformation: Spatial transformation of the label image, a dictionary containing the transformation type
            and parameters `{"type": "affine", "parameters": […]}`
        dimension_separator: Separator placed between the dimensions of a chunk.
        overwrite: If True, replace any existing array or group with the given name.
        is_label: Whether the array is a label image and should be scaled without interpolation.
    """
    delayed = []
    array_paths = []
    array_key = "s0"
    delayed.append(
        array_to_zarr(
            array=array,
            array_key=array_key,
            group=group,
            chunks=chunks,
            dimension_separator=dimension_separator,
            overwrite=overwrite,
            dask_compute=False,
        )
    )

    array_paths.append(array_key)

    # Create pyramids
    downscaler = ngff_spatially_rescale
    downscaler_kwargs = dict(scale=0.5, axes_names=dimension_axes, is_label=is_label)
    downscaler_name = f"{downscaler.__module__}.{downscaler.__name__}"
    for ii in range(1, n_scales):
        array = downscaler(array, **downscaler_kwargs)
        array_key = f"s{ii}"
        delayed.append(
            array_to_zarr(
                array=array,
                array_key=array_key,
                group=group,
                chunks=chunks,
                dimension_separator=dimension_separator,
                overwrite=overwrite,
                dask_compute=False,
            )
        )
        array_paths.append(array_key)
    # We are using delayed computation so that for the last pyramid level previous levels don't need
    # to be recomputed.
    da.compute(*delayed)

    # Metadata
    add_multiscales_metadata(
        group=group,
        array_paths=array_paths,
        dimension_axes=dimension_axes,
        downscaler_name=downscaler_name,
        downscaler_kwargs=downscaler_kwargs,
        downscaler_type="gaussian",
        transformation=transformation,
    )


def add_multiscales_metadata(
    group: zarr.Group,
    array_paths: Sequence[str],
    dimension_axes: Sequence[DimensionAxisType],
    name: Optional[str] = None,
    downscaler_name: Optional[str] = None,
    downscaler_version: Optional[str] = None,
    downscaler_kwargs: Optional[dict] = None,
    downscaler_type: Optional[Literal["gaussian", "laplacian", "reduce", "pick"]] = None,
    transformation: Optional[dict] = None,
):
    metadata = [
        {
            "version": NGFF_SPEC_VERSION,
            "name": name if name is not None else downscaler_type,
            "axes": dimension_axes,
            "datasets": [{"path": name} for name in array_paths],
            "type": downscaler_type,
            "metadata": {
                "method": downscaler_name,
                "version": downscaler_version,
                # "args": "TODO",
                "kwargs": downscaler_kwargs,
            },
        }
    ]
    if transformation is not None:
        metadata[0]["_transformation"] = transformation
    group.attrs["multiscales"] = metadata
    for k in group.array_keys():
        group[k].attrs["_ARRAY_DIMENSIONS"] = dimension_axes


def add_omero_metadata(
    group: zarr.Group,
    channel_names: Optional[Sequence[str]] = None,
    channel_metadata: Optional[Sequence[dict]] = None,
):
    if channel_names is not None or channel_metadata is not None:
        if channel_names is None:
            channel_names = [None] * len(channel_metadata)
        elif channel_metadata is None:
            channel_metadata = [dict()] * len(channel_names)
        metadata = {
            # "id": 1, # DUMMY data
            # "name": name, # DUMMY data
            # "version": NGFF_SPEC_VERSION,
            "channels": [
                {
                    # TODO: These are dummy data, minimally required properties to prevent Napari crash
                    "active": True,
                    # "coefficient": 1,
                    "color": "FFFFFF",
                    "family": "linear",
                    "inverted": False,
                    "window": {
                        # "end": 1500,
                        # "max": 65535,
                        # "min": 0,
                        # "start": 0
                    },
                    **metadata,
                    "label": channel_name,
                }
                for channel_name, metadata in zip(channel_names, channel_metadata)
            ]
        }
        group.attrs["omero"] = metadata


def add_labels_metadata(group: zarr.Group, data_path: str):
    metadata = group.attrs.get("labels", [])
    if data_path not in metadata:
        metadata.append(data_path)
    group.attrs["labels"] = sorted(set(metadata))


def add_label_metadata(
    group: zarr.Group,
    colors: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    properties: Optional[Sequence[dict]] = None,
):
    metadata = {"version": NGFF_SPEC_VERSION}
    if colors is not None or properties is not None:
        label_image_unscaled = group[get_first_array_key(group)]
        labels = np.unique(label_image_unscaled)[1:]
        if colors is not None:
            assert len(colors) == len(labels)
            assert all(
                isinstance(color, Sequence)
                and len(color) == 4
                and all(isinstance(c, int) and 0 <= c < 256 for c in color)
                for color in colors
            )
            metadata["colors"] = [
                [{"label-value": label, "rgba": color} for label, color in zip(labels, colors)]
            ]
        if properties is not None:
            assert len(properties) == len(labels)
            metadata["properties"] = [
                [{"label-value": label, **props} for label, props in zip(labels, properties)]
            ]
    group.attrs["image-label"] = metadata


def add_collection_metadata(
    group: zarr.Group,
    collection_name: str,
    image_path: Optional[str] = None,
    image_properties: Optional[dict] = None,
):
    if image_properties is None:
        image_properties = {}
    collections = group.attrs.get("_collections", [])
    names = [collection.get("name") for collection in collections]
    if collection_name in names:
        index = names.index(collection_name)
        collection = collections[index]
    else:
        collection = {"name": collection_name, "images": {}}
        collections.append(collection)
    if image_path is not None:
        collection["images"][image_path] = image_properties
    group.attrs["_collections"] = collections


def get_channel_index(group: zarr.Group, channel_name: str) -> int:
    assert "omero" in group.attrs
    channel_metadatas = group.attrs["omero"].get("channels")
    assert channel_metadatas is not None
    for index, channel_metadata in enumerate(channel_metadatas):
        if channel_metadata.get("label") == channel_name:
            return index
    raise RuntimeError(f"Channel name {channel_name} not found")


def get_first_array_key(group: zarr.Group) -> str:
    return list(group.array_keys())[0]


def get_chunks(shape: Tuple[int, ...], axes_names: Sequence[DimensionAxisType]) -> Tuple[int, ...]:
    assert len(shape) == len(axes_names)
    used_axes = [axis for axis, dim in zip(axes_names, shape) if dim > 1]
    spatial_axes = set(used_axes) & SPATIAL_DIMENSIONS

    chunksize_spatial = 256 if len(spatial_axes) < 3 else 64
    chunksize_other = 1
    return tuple(
        chunksize_spatial if name in spatial_axes else chunksize_other for name in axes_names
    )


def assert_multiscale_same_values(group: zarr.Group, array_paths: Sequence[str]):
    """
    Assert that all pyramid levels have the same set of values (no interpolation or smooth sampling).

    Args:
        group: The Zarr group containing pyramids
        array_paths: The paths / keys of the pyramid arrays, in order of decreasing scale.
    """
    # TODO: If array path is not a key (but folder path), can it still be accessed like that?
    array_orig = group[array_paths[0]]
    expected = set(np.unique(array_orig))
    for i in range(1, len(array_paths)):
        array_level = group[array_paths[i]]
        found = set(np.unique(array_level))
        if not expected.issuperset(found):
            raise AssertionError(
                f"{len(found)} found values are not " "a subset of {len(expected)} values"
            )


def assert_omero_channel_names(image_group: zarr.Group, n_channels: int):
    if image_group.attrs.get("omero", {}).get("channels"):
        n_omero_channels = len(image_group.attrs["omero"]["channels"])
        assert n_omero_channels == n_channels


def array_to_zarr(
    array: Union[np.ndarray, da.Array],
    array_key: str,
    group: zarr.Group,
    chunks: Sequence[int],
    dimension_separator: DimensionSeparatorType,
    overwrite: bool,
    dask_compute: bool = True,
) -> Optional[Delayed]:
    """
    Write an array to a Zarr store. Drop-in replacement for group.create_dataset supporting Dask.

    Args:
        array: A Numpy or Dask array to write to Zarr
        array_key: The key under which to store the array
        group: The Zarr group
        chunks: Chunk shape. If not provided, will be guessed from `shape` and `dtype`.
        dimension_separator: Separator placed between the dimensions of a chunk.
        overwrite: If True, replace any existing array or group with the given name.
        dask_compute: If a dask array is passed and compute is False, returns a dask Delayed object
            that will save the array to zarr later when it is explicitely computed.

    Returns:
        A dask Delayed object if a dask array is passed and compute is False.
    """
    if isinstance(array, da.Array):
        delayed = da.to_zarr(
            arr=da.array(array).rechunk(chunks=chunks),
            url=group.store,
            component=str(Path(group.path, array_key)),
            dimension_separator=dimension_separator,
            overwrite=overwrite,
            compute=dask_compute,
        )
        if not dask_compute:
            return delayed
    elif isinstance(array, np.ndarray):
        group.create_dataset(
            name=array_key,
            data=array,
            chunks=chunks,
            dimension_separator=dimension_separator,
            overwrite=overwrite,
        )
    else:
        raise (TypeError(f"Array must be a np.ndarray or dask.array.Array, not {type(array)}"))
