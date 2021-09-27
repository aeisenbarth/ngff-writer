from pathlib import Path

import numpy as np
import pytest
from pkg_resources import resource_filename

from ngff_writer.writer import read_from_ngff_zarr

np.random.seed(1)


@pytest.fixture
def spacem_mini_dataset4x5_zarr_path():
    return Path(resource_filename("tests", "resources/spacem_mini_dataset4x5.zarr")).resolve()


def test_read_image_from_ngff_zarr(spacem_mini_dataset4x5_zarr_path):
    actual = read_from_ngff_zarr(
        store=str(spacem_mini_dataset4x5_zarr_path), collection="well1", image="post_maldi"
    )
    assert actual.name == "/well1/post_maldi/s0"


def test_read_image_from_ngff_zarr_without_collection(spacem_mini_dataset4x5_zarr_path):
    actual = read_from_ngff_zarr(store=str(spacem_mini_dataset4x5_zarr_path / "well1/post_maldi"))
    assert actual.name == "/s0"


def test_read_label_from_ngff_zarr(spacem_mini_dataset4x5_zarr_path):
    actual = read_from_ngff_zarr(
        store=str(spacem_mini_dataset4x5_zarr_path),
        collection="well1",
        image="post_maldi",
        label="ablation_marks",
    )
    assert actual.name == "/well1/post_maldi/labels/ablation_marks/s0"


def test_read_attribute_from_ngff_zarr(spacem_mini_dataset4x5_zarr_path):
    actual = read_from_ngff_zarr(
        store=str(spacem_mini_dataset4x5_zarr_path),
        collection="well1",
        image="post_maldi",
        attribute="multiscales",
    )
    assert isinstance(actual, list)
    assert len(actual) == 1
    assert isinstance(actual[0], dict)
    assert actual[0]["datasets"][0] == {"path": "s0"}


def test_read_nested_attribute_from_ngff_zarr(spacem_mini_dataset4x5_zarr_path):
    actual = read_from_ngff_zarr(
        store=str(spacem_mini_dataset4x5_zarr_path),
        collection="well1",
        image="post_maldi",
        attribute=["multiscales", 0, "datasets", 0, "path"],
    )
    assert actual == "s0"
