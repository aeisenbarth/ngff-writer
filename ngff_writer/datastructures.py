import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import zarr
from dask import array as da

from ngff_writer.constants import DIMENSION_SEPARATOR, ZARR_DISALLOWED_CHARS_REGEX
from ngff_writer.json_utils import nested_dict_apply_fn_at_path
from ngff_writer.typing import DimensionAxisType, DimensionSeparatorType
from ngff_writer.writer_utils import (
    add_collection_metadata,
    add_image,
    add_label,
    get_channel_index,
    set_image_array,
    set_label_array,
)
from ngff_writer.array_utils import select_dimensions


class _NgffBase:
    def __init__(
        self,
        overwrite: bool = False,
        dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
    ):
        """
        Base class for NGFF datastructures

        Args:
            overwrite: If True, replace any existing array or group with the given name.
            dimension_separator: Separator placed between the dimensions of a chunk.
        """
        self.overwrite = overwrite
        self.dimension_separator = dimension_separator


class _NgffImageBase(_NgffBase):
    def __init__(self, group: zarr.Group, name: Optional[str] = None, **kwargs):
        """
        Base class for image-like NGFF datastructures

        Args:
            group: The Zarr group which corresponds to the image and contains its data arrays
            name: A name for the image. Used by images contained in a collection of labels.
                Otherwise None.
        """
        super().__init__(**kwargs)
        self.group = group
        self.name = name

    def array(
        self,
        level: int = 0,
        time_point: Optional[int] = None,
        channel_name: Optional[str] = None,
        dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
        as_dask: bool = False,
    ) -> Union[zarr.Array, da.Array]:
        """
        Returns a Zarr array of the image.

        Args:
            level: The scale level of the image pyramid. Returns by default the highest resolution.
            time_point: A time_point to select from the array. If dimension_axes is not given,
                a 5D array is returned.
            channel_name: A channel to slice from the array. If dimension_axes is not given,
                a 5D array is returned.
            dimension_axes: Sequence of dimension names to slice from the array. The array will be
                converted to a Numpy array (if not `as_dask` is used) and only contains these
                dimensions in the specified order.
            as_dask: Whether to return a dask array. If False, a Zarr array is returned if possible.

        Returns:
            An array
        """
        assert len(self.group.attrs.get("multiscales", [])) >= 1
        multiscale_metadata = self.group.attrs["multiscales"][0]
        assert len(multiscale_metadata.get("datasets", [])) >= 1
        dataset_paths = [dct["path"] for dct in multiscale_metadata["datasets"]]
        assert level < len(dataset_paths), f"Image does not have a pyramid for level {level}"
        dataset_path = dataset_paths[level]
        array = self.group[dataset_path]
        if as_dask:
            # Don't use from_dask directly with path/url because it may not support all of zarr's
            # options. For example https://github.com/dask/dask/issues/7673
            array = da.from_zarr(array)
        if time_point is not None:
            array = array[time_point : time_point + 1, :, :, :, :]
        if channel_name is not None:
            channel_index = get_channel_index(self.group, channel_name)
            array = array[:, channel_index : channel_index + 1, :, :, :]
        if dimension_axes is not None:
            all_dimension_axes = self.attribute(["multiscales", 0, "axes"])
            array = select_dimensions(array, dimension_axes, all_dimension_axes)
        return array

    @abstractmethod
    def set_array(
        self,
        array: Union[np.ndarray, da.Array],
        dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
        chunks: Optional[Sequence[int]] = None,
        n_scales: Optional[int] = None,
        transformation: Optional[dict] = None,
        overwrite: Optional[bool] = None,
    ):
        """
        Sets or replaces the data array of the image

        Args:
            array: The new image data array as 5D-array (tczyx)
            dimension_axes: Optional sequence of used dimension axes. Currently all are required in
                tczyx order.
            chunks: Optional chunk shape. If not provided, will be guessed from `shape` and `dtype`.
            n_scales: Optional number of scales (including unscaled) arrays in the pyramid
            transformation: Spatial transformation of the label image, a dictionary containing the
                transformation type and parameters `{"type": "affine", "parameters": […]}`
            overwrite: If True, replace any existing array or group with the given name.
        """
        pass

    @property
    def channel_names(self) -> List[str]:
        """
        Returns a list of channel names. Raises an error if no names are defined. The order of the
        names corresponds to the C dimension of the array.
        """
        assert "omero" in self.group.attrs
        omero_metadata = self.group.attrs["omero"]
        assert "channels" in omero_metadata
        return [dct.get("label") for dct in omero_metadata["channels"]]

    @property
    def transformation(self) -> Optional[np.ndarray]:
        """
        Returns the transformation parameters if a transformation is defined, otherwise None.
        To be used for initializing a transformation object.
        """
        assert len(self.group.attrs.get("multiscales", [])) >= 1
        multiscale_metadata = self.group.attrs["multiscales"][0]
        transformation = multiscale_metadata.get("_transformation")
        if transformation is not None and transformation["type"] == "affine":
            return multiscale_metadata["_transformation"]["parameters"]

    def attribute(self, key: Union[str, Sequence[Union[str, int]]], default: Any = KeyError) -> Any:
        """
        Returns an attribute value.

        Args:
            key: A string representing a top-level attribute or a sequence of strings for a nested
                attribute.

        Returns:
            The attribute's value
        """
        if isinstance(key, str):
            return self.group.attrs[key]
        elif isinstance(key, Sequence) and all(isinstance(a, (str, int)) for a in key):
            try:
                return nested_dict_apply_fn_at_path(
                    nested_dict=self.group.attrs,
                    key_path=key,
                    fn=lambda sub_dct, key: sub_dct[key],
                    create=False,
                )
            except KeyError as e:
                if default == KeyError:
                    raise e
                else:
                    return default
        else:
            raise TypeError("Argument 'key' must be a string or a list of strings")


class NgffLabel(_NgffImageBase):
    def __init__(
        self,
        group: zarr.Group,
        name: Optional[str] = None,
        labels_group: Optional[zarr.Group] = None,
        image_group: Optional[zarr.Group] = None,
        **kwargs,
    ):
        """
        Wrapper class around a NGFF label image

        Args:
            group: The Zarr group which corresponds to the label image and contains its data arrays
            name: A name for the labels
            labels_group: The Zarr group containing all label images.
            image_group: The Zarr group of the image to which the label image refers to.
        """
        super().__init__(group=group, name=name, **kwargs)
        self.labels_group = labels_group
        self.image_group = image_group

    def set_array(
        self,
        array: Union[np.ndarray, da.Array],
        dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
        chunks: Optional[Sequence[int]] = None,
        n_scales: Optional[int] = None,
        transformation: Optional[dict] = None,
        overwrite: Optional[bool] = None,
    ):
        set_label_array(
            array=array,
            label_group=self.group,
            labels_group=self.labels_group,
            image_group=self.image_group,
            dimension_axes=dimension_axes,
            chunks=chunks,
            n_scales=n_scales,
            transformation=transformation,
            overwrite=overwrite if overwrite is not None else self.overwrite,
            dimension_separator=self.dimension_separator,
        )


class NgffImage(_NgffImageBase):
    def __init__(self, group: zarr.Group, name: Optional[str] = None, **kwargs):
        """
        Wrapper class around a NGFF image

        Args:
            group: The Zarr group which corresponds to the image and contains its data arrays
            name: A name for the image
        """
        super().__init__(group=group, name=name, **kwargs)

    def set_array(
        self,
        array: Union[np.ndarray, da.Array],
        dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
        chunks: Optional[Sequence[int]] = None,
        n_scales: Optional[int] = None,
        transformation: Optional[dict] = None,
        channel_names: Optional[Sequence[str]] = None,
        channel_metadata: Optional[Sequence[dict]] = None,
        overwrite: Optional[bool] = None,
    ):
        set_image_array(
            array=array,
            image_group=self.group,
            dimension_axes=dimension_axes,
            chunks=chunks,
            n_scales=n_scales,
            transformation=transformation,
            channel_names=channel_names,
            channel_metadata=channel_metadata,
            overwrite=overwrite if overwrite is not None else self.overwrite,
            dimension_separator=self.dimension_separator,
        )

    @property
    def labels(self) -> Dict[str, NgffLabel]:
        if not self.group.get("labels"):
            # Image does not have any labels
            return {}
        labels_group = self.group["labels"]
        assert "labels" in labels_group.attrs
        label_paths = self.group["labels"].attrs["labels"]
        return {
            label_path: NgffLabel(
                group=self.group["labels"][label_path],
                name=label_path,
                labels_group=labels_group,
                image_group=self.group,
            )
            for label_path in label_paths
        }

    def add_label(
        self,
        name: Optional[str] = None,
        array: Union[np.ndarray, da.Array] = None,
        dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
        chunks: Optional[Sequence[int]] = None,
        n_scales: Optional[int] = None,
        transformation: Optional[dict] = None,
        colors: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        properties: Optional[Sequence[dict]] = None,
    ) -> NgffLabel:
        """
        Adds a label image to the NGFF-Zarr file.

        Args:
            name: The name for the label
            array: The label image data as 5D-array (tczyx)
            dimension_axes: Optional sequence of used dimension axes. Currently all are required in
                tczyx order.
            chunks: Optional chunk shape. If not provided, will be guessed from `shape` and `dtype`.
            n_scales: Optional number of scales (including unscaled) arrays in the pyramid
            transformation: Spatial transformation of the label image, a dictionary containing the
                transformation type and parameters `{"type": "affine", "parameters": […]}`
            colors: Optional list of Hexadecimal color codes for each label value, in the order of
                the actually occurring label values.
            properties: Optional list of dictionaries of properties for each label value.

        Returns:
            The Zarr group of the added label
        """
        return add_label(
            group=self.group,
            array=array,
            name=name,
            dimension_axes=dimension_axes,
            chunks=chunks,
            n_scales=n_scales,
            transformation=transformation,
            colors=colors,
            properties=properties,
            overwrite=self.overwrite,
            dimension_separator=self.dimension_separator,
        )


class NgffCollection(_NgffBase):
    """
    Wrapper class around a NGFF collection
    """

    def __init__(
        self,
        group: zarr.Group,
        images: Dict[str, dict] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The group containing the collections, since a single collection is not bound to a group
        self.group = group
        self.name = name or ""
        # A dictionary containing image paths relative to the collections group
        self._images = images if images is not None else {}
        self._kwargs = kwargs

    @property
    def images(self) -> Dict[str, NgffImage]:
        return {
            Path(image_path).name: NgffImage(
                group=self.group[image_path], name=Path(image_path).name, **self._kwargs
            )
            for image_path in self._images.keys()
        }

    def add_image(
        self,
        image_name: str,
        array: Union[np.ndarray, da.Array] = None,
        dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
        chunks: Optional[Sequence[int]] = None,
        channel_names: Optional[Sequence[str]] = None,
        # TODO: channel_metadata: Maybe better dict with channel names as keys?
        channel_metadata: Optional[Sequence[dict]] = None,
        n_scales: Optional[int] = None,
        transformation: Optional[dict] = None,
    ) -> NgffImage:
        """
        Adds an image to the NGFF collection.

        Args:
            image_name: Name of the image
            array: The label image data as 5D-array (tczyx)
            dimension_axes: Optional sequence of used dimension axes. Currently all are required in
                tczyx order.
            chunks: Optional chunk shape. If not provided, will be guessed from `shape` and `dtype`.
            channel_names: Index for the "c" axis
            channel_metadata: Optional additional metadata for each channel.
            n_scales: Optional number of scales (including unscaled) arrays in the pyramid
            transformation: Spatial transformation of the label image, a dictionary containing the
                transformation type and parameters `{"type": "affine", "parameters": […]}`

        Returns:
            Instance of a wrapper around NGFF-Zarr image
        """
        assert not re.match(ZARR_DISALLOWED_CHARS_REGEX, image_name)
        image_path = str(Path(self.name, image_name))
        image_group = self.group.require_group(image_path, overwrite=self.overwrite)
        ngff_image = add_image(
            image_group,
            array=array,
            dimension_axes=dimension_axes,
            chunks=chunks,
            channel_names=channel_names,
            channel_metadata=channel_metadata,
            n_scales=n_scales,
            transformation=transformation,
            overwrite=self.overwrite,
            dimension_separator=self.dimension_separator,
        )
        add_collection_metadata(self.group, collection_name=self.name, image_path=image_path)
        return ngff_image


class NgffCollections:
    """
    Wrapper class around an NGFF group with image collections
    """

    def __init__(self, group: zarr.Group, **kwargs):
        self.group = group
        self._collection_kwargs = kwargs

    @property
    def collections(self) -> Dict[str, NgffCollection]:
        # TODO: "name" is not required, so the returned dictionary cannot contain all collections
        #  with missing name (key None).
        return {
            dct["name"]: NgffCollection(
                group=self.group,
                name=dct.get("name"),
                images=dct["images"],
                **self._collection_kwargs,
            )
            for dct in self.group.attrs.get("_collections", [])
        }

    def add_collection(self, name: str) -> NgffCollection:
        """
        Adds a collection to the NGFF-Zarr file.

        Args:
            name: Name of the collection

        Returns:
            Instance of a wrapper around NGFF-Zarr collection
        """
        assert not re.match(ZARR_DISALLOWED_CHARS_REGEX, name)
        add_collection_metadata(self.group, collection_name=name)
        return NgffCollection(group=self.group, name=name, **self._collection_kwargs)


class NgffZarr(NgffCollections, NgffCollection, NgffImage, NgffLabel):
    """
    Wrapper class around any of NgffCollections, NgffCollection, NgffImage, NgffLabel
    It is used when a new NGFF-Zarr file is created and the type of content has not yet been defined.
    """

    def __init__(
        self,
        group=zarr.Group,
        overwrite: bool = False,
        dimension_separator: DimensionSeparatorType = DIMENSION_SEPARATOR,
    ):
        super().__init__(group=group, overwrite=overwrite, dimension_separator=dimension_separator)
        self.group = group
        self.overwrite = overwrite
        self.dimension_separator = dimension_separator

    @property
    def images(self) -> Dict[str, NgffImage]:
        image_keys = [k for k in self.group.group_keys() if self.group[k].attrs.get("multiscales")]
        return {
            image_key: NgffImage(
                group=self.group[image_key],
                name=image_key,
                overwrite=self.overwrite,
                dimension_separator=self.dimension_separator,
            )
            for image_key in image_keys
        }

    def add_image(
        self,
        image_name: str = None,
        array: Union[np.ndarray, da.Array] = None,
        dimension_axes: Optional[Sequence[DimensionAxisType]] = None,
        chunks: Optional[Sequence[int]] = None,
        channel_names: Optional[Sequence[str]] = None,
        # TODO: channel_metadata: Maybe better dict with channel names as keys?
        channel_metadata: Optional[Sequence[dict]] = None,
        n_scales: Optional[int] = None,
        transformation: Optional[dict] = None,
    ) -> NgffImage:
        """
        Adds an image to the NGFF-Zarr file.

        Args:
            image_name: If not None, interprets this Zarr as an NgffCollection and adds a new image
                in it, otherwise interprets it as an NgffImage and sets the image data.
            array: The label image data as 5D-array (tczyx)
            dimension_axes: Optional sequence of used dimension axes. Currently all are required in
                tczyx order.
            chunks: Optional chunk shape. If not provided, will be guessed from `shape` and `dtype`.
            channel_names: Index for the "c" axis
            channel_metadata: Optional additional metadata for each channel.
            n_scales: Optional number of scales (including unscaled) arrays in the pyramid
            transformation: Spatial transformation of the label image, a dictionary containing the
                transformation type and parameters `{"type": "affine", "parameters": […]}`

        Returns:
            Instance of a wrapper around NGFF-Zarr image
        """
        if image_name is not None:
            assert not re.match(ZARR_DISALLOWED_CHARS_REGEX, image_name)
            image_group = self.group.require_group(image_name, overwrite=self.overwrite)
        else:
            image_group = self.group
        ngff_image = add_image(
            image_group,
            array=array,
            dimension_axes=dimension_axes,
            chunks=chunks,
            channel_names=channel_names,
            channel_metadata=channel_metadata,
            n_scales=n_scales,
            transformation=transformation,
            overwrite=self.overwrite,
            dimension_separator=self.dimension_separator,
        )
        return ngff_image
