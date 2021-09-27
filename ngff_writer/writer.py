import contextlib
from pathlib import Path
from typing import Generator, MutableMapping, Optional, Sequence, Union

import zarr

from ngff_writer.constants import DIMENSION_SEPARATOR
from ngff_writer.datastructures import NgffCollections, NgffImage, NgffLabel, NgffZarr
from ngff_writer.typing import DimensionAxisType, DimensionSeparatorType, JsonType, ZarrModeType


@contextlib.contextmanager
def open_ngff_zarr(
    store: Union[str, Path, MutableMapping[str, bytes]],
    mode: ZarrModeType = "a",
    storage_options: Optional[dict] = None,
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
    overwrite: bool = False,
) -> Generator[NgffZarr, None, None]:
    """
    Creates a Zarr file and opens it for adding NGFF-Zarr image data.

    Args:
        store: Zarr store or path or URL to the Zarr file
        mode: File-mode-like, Persistence mode:
            'r' means read only (must exist);
            'r+' means read/write (must exist);
            'a' means read/write (create if doesn't exist);
            'w' means create (overwrite if exists);
            'w-' means create (fail if exists).
        storage_options: fsspec storage options for accessing the file system when a URL is used
        dimension_separator: Separator placed between the dimensions of a chunk.
        overwrite: If True, replace any existing array or group with the given name.

    Yields:
        An NgffZarr wrapper object which allows to add images.
    """
    if isinstance(store, Path):
        store = str(store)
    # Convert the path/url into a store object because when a string is passed zarr-python does not
    # detect dimension_separator. See https://github.com/zarr-developers/zarr-python/issues/530
    if isinstance(store, str):
        if storage_options is None:
            storage_options = {}
        if dimension_separator is not None:
            storage_options["key_separator"] = dimension_separator
        store = zarr.storage.FSStore(store, auto_mkdir=True, **storage_options)

    with zarr.open(store, mode=mode) as group:
        if "_collections" in group.attrs:
            yield NgffCollections(
                group=group, dimension_separator=dimension_separator, overwrite=overwrite
            )
        elif "multiscales" in group.attrs:
            if "labels" in dimension_separator.split(group.store.path):
                yield NgffLabel(
                    group=group, dimension_separator=dimension_separator, overwrite=overwrite
                )
            else:
                yield NgffImage(
                    group=group, dimension_separator=dimension_separator, overwrite=overwrite
                )
        else:
            # New file, return an object that can be used to create any type
            yield NgffZarr(
                group=group, dimension_separator=dimension_separator, overwrite=overwrite
            )


def read_from_ngff_zarr(
    store: Union[str, Path, MutableMapping[str, bytes]],
    storage_options: Optional[dict] = None,
    dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
    collection: Optional[str] = None,
    image: Optional[str] = None,
    label: Optional[str] = None,
    attribute: Optional[Union[str, Sequence[Union[str, int]]]] = None,
    level: int = 0,
    time_point: Optional[int] = None,
    channel_name: Optional[str] = None,
    dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
    as_dask: bool = False,
) -> Union[zarr.Array, JsonType]:
    """
    Utility method for reading a single item from an NGFF-Zarr file. The file reference is closed
    afterwards. If `attribute` is specified, the attribute value is returned, otherwise an array is
    returned.

    Args:
        store: Zarr store or path or URL to the Zarr file
        storage_options: fsspec storage options for accessing the file system when a URL is used
        dimension_separator: Separator placed between the dimensions of a chunk.
        collection: Name of the collection if the Zarr file contains collections
        image: Name of the image within the collection if the Zarr file contains collections
        label: Name of the label if the Zarr file contains an image
        attribute: Name of the label if the Zarr file contains an image
        level: The scale level of the image pyramid. Returns by default the highest resolution.
        time_point: A time_point to select from the array. If dimension_axes is not given, a 5D
             array is returned.
        channel_name: A channel to slice from the array. If dimension_axes is not given, a 5D array
            is returned.
        dimension_axes: Sequence of dimension names to slice from the array. The array will be
            converted to a Numpy array (if not `as_dask` is used) and only contains these dimensions
            in the specified order.
        as_dask: Whether to return a dask array. If False, a Zarr array is returned if possible.

    Returns:
        The specified array or attribute value.
    """
    if isinstance(store, Path):
        store = str(store)
    # Convert the path/url into a store object because when a string is passed zarr-python does not
    # detect dimension_separator. See https://github.com/zarr-developers/zarr-python/issues/530
    if isinstance(store, str):
        if storage_options is None:
            storage_options = {}
        if dimension_separator is not None:
            storage_options["key_separator"] = dimension_separator
        store = zarr.storage.FSStore(store, auto_mkdir=True, **storage_options)

    with zarr.open(store, storage_options=storage_options, mode="r") as group:
        if collection is not None:
            if image is None:
                raise TypeError("Argument 'image' required")
            _collection = NgffCollections(group=group).collections[collection]
            _image = _collection.images[image]
        else:
            _image = NgffImage(group=group[image] if image is not None else group)

        if label is not None:
            _label = _image.labels[label]
            if attribute is not None:
                return _label.attribute(attribute)
            else:
                return _label.array(
                    level=level,
                    time_point=time_point,
                    channel_name=channel_name,
                    dimension_axes=dimension_axes,
                    as_dask=as_dask,
                )
        elif attribute is not None:
            return _image.attribute(attribute)
        else:
            return _image.array(
                level=level,
                time_point=time_point,
                channel_name=channel_name,
                dimension_axes=dimension_axes,
                as_dask=as_dask,
            )
