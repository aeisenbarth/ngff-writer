import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import zarr
from pkg_resources import resource_filename
from skimage.transform import resize as ski_resize
from zarr.storage import FSStore

from ngff_writer.dask_utils import resize as dask_resize
from ngff_writer.writer import NgffImage, NgffZarr, open_ngff_zarr
from ngff_writer.writer_utils import (
    array_to_zarr,
)
from ngff_writer.array_utils import (
    apply_over_axes,
    ngff_spatially_rescale,
    select_dimensions,
    to_tczyx,
)
from ngff_writer.constants import DIMENSION_AXES

np.random.seed(1)


@pytest.fixture
def array(t: int = 1, c: int = 3, z: int = 1, y: int = 16, x: int = 16) -> np.ndarray:
    shape = (t, c, z, y, x)
    return (np.random.random(shape) * 255).astype(np.uint)


@pytest.fixture
def label_array(
    t: int = 1, c: int = 1, z: int = 1, y: int = 16, x: int = 16, n_labels: int = 4
) -> np.ndarray:
    shape = (t, c, z, y, x)
    ndim = sum(1 for s in shape if s > 1)
    r = (0.2 * np.prod(shape) ** (1 / ndim)).astype(int)
    array = np.zeros(shape, dtype=np.uint)
    for label in range(1, n_labels + 1):
        center = np.random.randint(0, shape)
        lower = np.maximum(0, center - r)
        upper = np.minimum(center + r, shape)
        slices = tuple(slice(l, u) for l, u in zip(lower, upper))
        array[slices] = label
    return array


@pytest.fixture
def zarr_group(tmpdir: str):
    path = Path(tmpdir.mkdir("data"))
    store = FSStore(
        str(path.resolve()), mode="w", dimension_separator="/", normalize_keys=True, auto_mkdir=True
    )
    root = zarr.group(store=store)
    group = root.create_group("test")
    return group


@pytest.fixture
def ngff_zarr(zarr_group: zarr.Group):
    return NgffZarr(zarr_group, dimension_separator="/", overwrite=False)


@pytest.fixture
def ngff_image(ngff_zarr, array: np.ndarray):
    return ngff_zarr.add_image(array=array)


@pytest.fixture
def spacem_mini_dataset4x5_path():
    return Path(resource_filename("tests", "resources/spacem_mini_dataset4x5")).resolve()


def test_numpy_array_to_zarr(zarr_group: zarr.Group, array: np.ndarray):
    np_array = np.array(array)
    array_key = "some_key"
    array_to_zarr(
        array=np_array,
        array_key=array_key,
        group=zarr_group,
        chunks=(1, 1, 1, 4, 4),
        dimension_separator="/",
        overwrite=True,
    )
    expected = array
    actual = zarr_group[array_key]
    np.testing.assert_equal(actual, expected)


def test_dask_array_to_zarr(zarr_group: zarr.Group, array: np.ndarray):
    da_array = da.from_array(array, name="array", chunks=array.shape)
    array_key = "some_key"
    array_to_zarr(
        array=da_array,
        array_key="some_key",
        group=zarr_group,
        chunks=(1, 1, 1, 4, 4),
        dimension_separator="/",
        overwrite=True,
    )
    expected = array
    actual = zarr_group[array_key]
    np.testing.assert_equal(actual, expected)


def test_select_dimensions_with_numpy():
    axes, shape = zip(("t", 1), ("c", 3), ("z", 1), ("y", 16), ("x", 16))
    array = np.arange(np.prod(shape)).reshape(shape)
    actual = select_dimensions(array, ("y", "x", "c"), all_dimension_axes=axes)
    expected = np.dstack(
        [
            np.arange(16 ** 2).reshape((16, 16)),
            np.arange(16 ** 2, 2 * 16 ** 2).reshape((16, 16)),
            np.arange(2 * 16 ** 2, 3 * 16 ** 2).reshape((16, 16)),
        ]
    )
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_equal(actual, expected)


def test_select_dimensions_with_dask():
    axes, shape = zip(("t", 1), ("c", 3), ("z", 1), ("y", 16), ("x", 16))
    array = np.arange(np.prod(shape)).reshape(shape)
    array_da = da.from_array(array)
    actual_da = select_dimensions(array_da, ("y", "x", "c"), all_dimension_axes=axes)
    actual = actual_da.compute()
    expected = np.dstack(
        [
            np.arange(16 ** 2).reshape((16, 16)),
            np.arange(16 ** 2, 2 * 16 ** 2).reshape((16, 16)),
            np.arange(2 * 16 ** 2, 3 * 16 ** 2).reshape((16, 16)),
        ]
    )
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_equal(actual, expected)


def test_write_ngff_zarr(tmpdir: str):
    expected_path = Path(tmpdir) / "test"
    with open_ngff_zarr(store=expected_path) as ngff_zarr:
        group: zarr.Group = ngff_zarr.group
        actual_path = group.store.path
        assert actual_path == str(expected_path)
        assert group.name == "/"


def test_add_image_from_numpy(ngff_zarr: NgffZarr, array: np.ndarray):
    n_scales = 2
    channel_names = ["brightfield", "GFP", "DAPI"]
    ngff_zarr.add_image(array=array, n_scales=n_scales, channel_names=channel_names)
    expected_array = array
    group = ngff_zarr.group
    # multiscales
    multiscales = group.attrs.get("multiscales")
    assert multiscales is not None
    assert len(multiscales[0]["datasets"]) == n_scales
    paths = [d["path"] for d in multiscales[0]["datasets"]]
    assert all(p in group.array_keys() for p in paths)
    actual_array_0 = group[paths[0]]
    actual_array_1 = group[paths[1]]
    np.testing.assert_equal(actual_array_0, expected_array)
    assert actual_array_1.shape == tuple(
        np.maximum(np.array([1, 1, 0.5, 0.5, 0.5]) * expected_array.shape, 1)
    )
    # omero
    omero = group.attrs.get("omero")
    assert omero is not None
    actual_channel_names = [c.get("label") for c in omero["channels"]]
    assert actual_channel_names == channel_names
    assert all(c.get("color") is not None for c in omero["channels"])
    assert actual_channel_names == channel_names


def test_add_image_from_dask(ngff_zarr: NgffZarr, array: np.ndarray):
    expected_array = array
    array: da.Array = da.from_array(array)
    n_scales = 2
    channel_names = ["brightfield", "GFP", "DAPI"]
    ngff_zarr.add_image(array=array, n_scales=n_scales, channel_names=channel_names)
    group = ngff_zarr.group
    # multiscales
    multiscales = group.attrs.get("multiscales")
    assert multiscales is not None
    assert len(multiscales[0]["datasets"]) == n_scales
    paths = [d["path"] for d in multiscales[0]["datasets"]]
    assert all(p in group.array_keys() for p in paths)
    actual_array_0 = group[paths[0]]
    actual_array_1 = group[paths[1]]
    np.testing.assert_equal(actual_array_0, expected_array)
    assert actual_array_1.shape == tuple(
        np.maximum(np.array([1, 1, 0.5, 0.5, 0.5]) * expected_array.shape, 1)
    )
    # omero
    omero = group.attrs.get("omero")
    assert omero is not None
    actual_channel_names = [c.get("label") for c in omero["channels"]]
    assert actual_channel_names == channel_names
    assert all(c.get("color") is not None for c in omero["channels"])
    assert actual_channel_names == channel_names


def test_add_label(ngff_image: NgffImage, label_array: np.ndarray):
    n_scales = 2
    label_group1 = ngff_image.add_label(name="segmentation1", array=label_array, n_scales=n_scales)
    group = ngff_image.group
    # labels
    assert "labels" in group.group_keys()
    labels_group = group.require_group("labels")
    assert "labels" in labels_group.attrs
    assert "segmentation1" in labels_group.attrs["labels"]
    assert "segmentation1" in labels_group.group_keys()
    assert label_group1.group == labels_group.require_group("segmentation1")
    # multiscales
    expected_array = label_array
    multiscales = label_group1.group.attrs.get("multiscales")
    assert multiscales is not None
    assert len(multiscales[0]["datasets"]) == n_scales
    paths = [d["path"] for d in multiscales[0]["datasets"]]
    assert all(p in label_group1.group.array_keys() for p in paths)
    actual_array_0 = label_group1.group[paths[0]]
    actual_array_1 = label_group1.group[paths[1]]
    np.testing.assert_equal(actual_array_0, expected_array)
    assert actual_array_1.shape == tuple(
        np.maximum(np.array([1, 1, 0.5, 0.5, 0.5]) * expected_array.shape, 1)
    )


def test_ngff_spatially_rescale(label_array):
    shape_yx = (16, 16)
    image_yx = np.arange(np.prod(shape_yx)).reshape(shape_yx)
    actual_yx = ngff_spatially_rescale(image_yx, scale=0.5, axes_names=("y", "x"))
    np.testing.assert_equal(actual_yx.shape, (8, 8))

    actual_yx = ngff_spatially_rescale(image_yx, scale=0.75, axes_names=("y", "x"))
    np.testing.assert_equal(actual_yx.shape, (12, 12))

    actual_yx = ngff_spatially_rescale(image_yx, scale=2.0, axes_names=("y", "x"))
    np.testing.assert_equal(actual_yx.shape, (32, 32))

    shape_zyx = (16, 16, 16)
    image_zyx = np.arange(np.prod(shape_zyx)).reshape(shape_zyx)
    actual_zyx = ngff_spatially_rescale(image_zyx, scale=0.5, axes_names=("z", "y", "x"))
    np.testing.assert_equal(actual_zyx.shape, (8, 8, 8))

    shape_cyx = (3, 16, 16)
    image_cyx = np.arange(np.prod(shape_cyx)).reshape(shape_cyx)
    actual_cyx = ngff_spatially_rescale(image_cyx, scale=0.5, axes_names=("c", "y", "x"))
    np.testing.assert_equal(actual_cyx.shape, (3, 8, 8))

    shape_yxc = (16, 16, 3)
    image_yxc = np.arange(np.prod(shape_yxc)).reshape(shape_yxc)
    actual_yxc = ngff_spatially_rescale(image_yxc, scale=0.5, axes_names=("y", "x", "c"))
    np.testing.assert_equal(actual_yxc.shape, (8, 8, 3))

    shape_tczyx = (1, 3, 1, 16, 16)
    image_tczyx = np.arange(np.prod(shape_tczyx)).reshape(shape_tczyx)
    actual_tczyx = ngff_spatially_rescale(image_tczyx, scale=0.5, axes_names=DIMENSION_AXES)
    np.testing.assert_equal(actual_tczyx.shape, (1, 3, 1, 8, 8))

    actual_label_array = ngff_spatially_rescale(
        label_array, scale=0.5, axes_names=DIMENSION_AXES, is_label=True
    )
    assert set(np.unique(actual_label_array)) <= set(np.unique(label_array))


def test_to_tczyx():
    axes, shape = zip(("t", 1), ("c", 3), ("z", 1), ("y", 16), ("x", 16))
    array = np.dstack(
        [
            np.arange(16 ** 2).reshape((16, 16)),
            np.arange(16 ** 2, 2 * 16 ** 2).reshape((16, 16)),
            np.arange(2 * 16 ** 2, 3 * 16 ** 2).reshape((16, 16)),
        ]
    )
    actual = to_tczyx(array, ("y", "x", "c"))
    expected = np.arange(np.prod(shape)).reshape(shape)
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_equal(actual, expected)


def test_ngff_spatially_rescale_with_dask(label_array):
    shape_yx = (16, 16)
    image_yx = np.arange(np.prod(shape_yx)).reshape(shape_yx)
    image_da_yx = da.from_array(image_yx)
    actual_yx = ngff_spatially_rescale(image_da_yx, scale=0.5, axes_names=("y", "x")).compute()
    np.testing.assert_equal(actual_yx.shape, (8, 8))

    # actual_yx = ngff_spatially_rescale(image_da_yx, scale=0.75, axes_names=("y", "x")).compute()
    # np.testing.assert_equal(actual_yx.shape, (12, 12))
    #
    # actual_yx = ngff_spatially_rescale(image_da_yx, scale=2.0, axes_names=("y", "x")).compute()
    # np.testing.assert_equal(actual_yx.shape, (32, 32))

    shape_zyx = (16, 16, 16)
    image_zyx = np.arange(np.prod(shape_zyx)).reshape(shape_zyx)
    image_da_zyx = da.from_array(image_zyx)
    actual_zyx = ngff_spatially_rescale(
        image_da_zyx, scale=0.5, axes_names=("z", "y", "x")
    ).compute()
    np.testing.assert_equal(actual_zyx.shape, (8, 8, 8))

    shape_cyx = (3, 16, 16)
    image_cyx = np.arange(np.prod(shape_cyx)).reshape(shape_cyx)
    image_da_cyx = da.from_array(image_cyx)
    actual_cyx = ngff_spatially_rescale(
        image_da_cyx, scale=0.5, axes_names=("c", "y", "x")
    ).compute()
    np.testing.assert_equal(actual_cyx.shape, (3, 8, 8))

    shape_yxc = (16, 16, 3)
    image_yxc = np.arange(np.prod(shape_yxc)).reshape(shape_yxc)
    image_da_yxc = da.from_array(image_yxc)
    actual_yxc = ngff_spatially_rescale(
        image_da_yxc, scale=0.5, axes_names=("y", "x", "c")
    ).compute()
    np.testing.assert_equal(actual_yxc.shape, (8, 8, 3))

    shape_tczyx = (1, 3, 1, 16, 16)
    image_tczyx = np.arange(np.prod(shape_tczyx)).reshape(shape_tczyx)
    image_da_tczyx = da.from_array(image_tczyx)
    actual_tczyx = ngff_spatially_rescale(
        image_da_tczyx, scale=0.5, axes_names=DIMENSION_AXES
    ).compute()
    np.testing.assert_equal(actual_tczyx.shape, (1, 3, 1, 8, 8))

    label_array_da = da.from_array(label_array)
    actual_label_array = ngff_spatially_rescale(
        label_array_da, scale=0.5, axes_names=DIMENSION_AXES, is_label=True
    ).compute()
    assert set(np.unique(actual_label_array)) <= set(np.unique(label_array))


def test_dask_resize():
    shape_tczyx = (1, 3, 1, 6, 6)
    expected_shape_tczyx = (1, 3, 1, 9, 9)
    image_tczyx = np.arange(np.prod(shape_tczyx)).reshape(shape_tczyx)
    image_da_tczyx = da.from_array(image_tczyx, chunks=(1, 1, 1, 3, 3))
    actual_tczyx = dask_resize(image_da_tczyx, expected_shape_tczyx).compute()
    np.testing.assert_equal(actual_tczyx.shape, expected_shape_tczyx)

    shape_tczyx = (1, 3, 1, 12, 12)
    expected_shape_tczyx = (1, 3, 1, 6, 6)
    image_tczyx = np.arange(np.prod(shape_tczyx)).reshape(shape_tczyx)
    image_da_tczyx = da.from_array(image_tczyx, chunks=(1, 1, 1, 3, 3))
    actual_tczyx = dask_resize(image_da_tczyx, expected_shape_tczyx).compute()
    np.testing.assert_equal(actual_tczyx.shape, expected_shape_tczyx)


def test_dask_resize_image(array: np.ndarray):
    array_yx_np = array[0, 0, 0, :, :]
    array_yx_da = da.from_array(array_yx_np)
    output_shape = (8, 8)
    actual = dask_resize(array_yx_da, output_shape, order=0, preserve_range=True).compute()
    expected = ski_resize(array_yx_np, output_shape, order=0, preserve_range=True).astype(
        array_yx_np.dtype
    )
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_equal(actual, expected)


def test_dask_resize_label(label_array: np.ndarray):
    array_yx_np = label_array[0, 0, 0, :, :]
    array_yx_da = da.from_array(array_yx_np)
    output_shape = (8, 8)
    actual = dask_resize(
        array_yx_da, output_shape, order=0, preserve_range=True, anti_aliasing=False
    ).compute()
    expected = ski_resize(
        array_yx_np, output_shape, order=0, preserve_range=True, anti_aliasing=False
    ).astype(array_yx_np.dtype)
    np.testing.assert_equal(actual.shape, expected.shape)
    assert set(np.unique(actual)) <= set(np.unique(array_yx_np))
    np.testing.assert_equal(actual, expected)


def test_apply_over_axes_with_numpy():
    expected_shape = (3, 4, 1, 4, 1)
    array = np.arange(np.prod(expected_shape)).reshape(expected_shape)
    func = lambda a: np.full(a.shape, np.mean(a))
    actual = apply_over_axes(func, array, (1, 3))
    np.testing.assert_equal(actual.shape, expected_shape)
    expected = (
        np.stack([np.full((4, 4), 7.5), np.full((4, 4), 23.5), np.full((4, 4), 39.5)])
        .astype(array.dtype)
        .reshape(expected_shape)
    )
    np.testing.assert_equal(actual, expected)


def test_apply_over_axes_with_dask():
    expected_shape = (3, 4, 1, 4, 1)
    array = np.arange(np.prod(expected_shape)).reshape(expected_shape)
    array_da = da.from_array(array)
    func = lambda a: da.full(a.shape, da.mean(a))
    actual_da = apply_over_axes(func, array_da, (1, 3))
    actual = actual_da.compute()
    np.testing.assert_equal(actual.shape, expected_shape)
    expected = (
        np.stack([np.full((4, 4), 7.5), np.full((4, 4), 23.5), np.full((4, 4), 39.5)])
        .astype(array.dtype)
        .reshape(expected_shape)
    )
    np.testing.assert_equal(actual, expected)


def test_bug_1():
    """
    When writing an NGFF-Zarr image with a small spatial dimension, the overlapping depth was too
    large.

    https://git.embl.de/grp-alexandrov/spacem-ht/-/issues/1
    """
    z = 2
    tmp = da.zeros((1, 1, z, 1000, 1000), chunks=(1, 1, 1, 128, 128))
    with tempfile.TemporaryDirectory() as temp_path:
        with open_ngff_zarr(Path(temp_path) / "tmp.zarr", overwrite=True) as f:
            f.add_image(array=tmp)
            # Should not raise ValueError
