# NGFF Writer

This is a higher-level wrapper around the Zarr library for creating and reading NGFF OME-Zarr datasets
for TCZYX image data.

 -  Compared to previous implementations, it uses object-oriented wrapper objects which allows to 
    build more complex NGFF images without using low-level Zarr functions.

 -  It not only accepts Numpy-like arrays, but also **Dask arrays**, which makes it possible to write
    really large datasets.

 -  It generates **pyramid levels**.

 -  It extends the NGFF specification 0.3 with most minimal implementations for:
  
    -  **transformations** (in private namespace as `_transformation` to avoid collisions with future 
       standards), following the proposal in [NGFF#28](https://github.com/ome/ngff/issues/28#issuecomment-786279835)
      
    -  **collections** (as `_collection`), following the proposal in 
       [NGFF#31](https://github.com/ome/ngff/issues/31#issuecomment-792582677)


## Installation

Create a new python environment for installing  

```
conda create -n ngff-writer python=3.8
conda activate ngff-writer
pip install git+https://github.com/aeisenbarth/ngff-writer
```

## Usage

### Creating an NGFF dataset

```python
import dask.array as da
import numpy as np
from dask_image.imread import imread

from ngff_writer.array_utils import affine_matrix_to_tczyx, to_tczyx
from ngff_writer.writer import open_ngff_zarr

with open_ngff_zarr(
    store="output.zarr",
    dimension_separator="/",
    overwrite=True,
) as f:
    channel_paths = ["img_t1_z1_c0.tif", "img_t1_z1_c1.tif", "img_t1_z1_c2.tif"]
    affine2d = np.array([[1.29, 0.12, 335.25], [-0.12, 1.29, 120.92], [0.0, 0.0, 1.0]])
    transformation = {
        "type": "affine",
        "parameters": affine_matrix_to_tczyx(affine2d, axes_names=("y", "x")).tolist(),
    }

    collection = f.add_collection(name="well1")

    image = collection.add_image(
        image_name="microscopy1",
        array=to_tczyx(
            da.concatenate(imread(p) for p in channel_paths), axes_names=("c", "y", "x")
        ),
        transformation=transformation,
        channel_names=["brightfield", "GFP", "DAPI"],
    )

    image.add_label(
        name="cells",
        array=to_tczyx(imread("cells.tif"), axes_names=("y", "x")),
        transformation=transformation,
    )
```

### Reading an NGFF dataset

Install additional packages:

```shell
pip install napari[all]==0.4.10
pip install git+https://github.com/aeisenbarth/ome-zarr-py@transformations-and-collections
pip install git+https://github.com/aeisenbarth/napari-ome-zarr@transformations-and-collections
```

Start Napari, which will load the plugins…

```shell
napari
```

…and drag&drop an NGFF dataset.

## Comments for discussion

 -  Generalizing **downscalers** or making downscalers customizable turned out to be complicated, 
    because there is a large combination of cases that need to be supported: Labels need a different 
    downscaler than images (no smooth sampling between labels), and for Dask and non-Dask arrays 
    different functions may need to be used. Moreover not all downscaling methods have a Dask 
    counterpart.
  
 -  Flexible order of dimension axes: If NGFF does not require all TCZYX axes in that order, readers
    have no guarantee and must check that a dataset contains certain dimensions or must reorder 
    dimensions. We noticed that reordering and inserting dimensions can lead to severe **performance**
    bottlenecks in Dask and algorithms first need to be optimized for that. Also, more complexity 
    can be a source of errors or can hold third-parties from fully adopting the specification.
    See helper methods [`to_tczyx`](./ngff_writer/array_utils.py), 
    [`select_dimensions`](./ngff_writer/array_utils.py) and [`apply_over_axes`](./ngff_writer/array_utils.py).

 -  Collections can contain more than one collection which can contain multiple images. However a 
    single collection does not exist as its own entity in the specification proposal used. That has
    several consequences:
  
    - It is not possible to directly load a **single collection** (it's not a Zarr group and not 
      required to be a directory separate from other collections) without loading all collections  
      and specifying the name of the desired collection.
      
    - Having loaded an image directly, it is not anymore possible to read the attributes dictionary 
      assigned to it in the collection, as that would require accessing a **parent** that may not exist.

 -  Transformations: This implementation uses an **affine matrix for 5-dimensions** (thus a 6×6 matrix), 
    which is the simplest approach ignoring that only some dimensions are actually spatial. 
    Since the Napari plugin splits channels of 5D arrays into separate layers (4D), the 
    channel dimension must also be stripped out of the matrix.
