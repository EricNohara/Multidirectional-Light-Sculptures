# Multidirectional Light Sculptures

## Eric Nohara-LeClair, ...

This code takes as input one or more reference silhouette images and generates a 3D printable voxel sculpture who's orthographic shadows match the inputs'. The pipeline creates the initial shadow hull, optimizes the input silhouettes to reduce inconsistencies to reduce missing parts of the silhouettes, then hollows out the interior of the sculpture to reduce the materials used to print it.

## Features

- Multi view shadow hull construction
- Iterative silhouette optimization to reduce missing shadow regions
- Fast greedy carving using per pixel support counts
- Hollow shell generation for material reduction
- Shadow simulation before fabrication
- Mesh export to STL format
- Debug visualizations including voxel slices and shadow comparisons
- Progress bars for long running optimization and carving stages

## Repository Structure

```bash
C:.
│   README.md
│
├───inputs
│       view0.png
│       view1.png
│       view4.png
│
├───outputs
│   ├───debug
│   │   ├───opt_iterations
│   │   └───slices
│   ├───meshes
│   └───sim
└───src
    │   carve.py
    │   config.py
    │   debug_slices.py
    │   deform.py
    │   distances.py
    │   export_mesh.py
    │   image_io.py
    │   optimize.py
    │   optimize_consistency.py
    │   projections.py
    │   render.py
    │   run_pipeline.py
    │   shadow_hull.py
    │   simulate.py
    │   voxel_ops.py
    │   warp.py
    │
    └───__pycache__
```

## Requirements

- Python 3.10 or higher
- numpy
- scipy
- tqdm
- trimesh
- pillow
- scikit image

### Install dependencies:

```bash
pip install numpy scipy tqdm trimesh pillow scikit-image
```

## Input Silhouettes

Silhouettes must satisfy the following:

- Binary images
- White pixels represent required shadow coverage
- Black pixels represent empty shadow
- All views must have the same image resolution

Place silhouette images inside the inputs folder and configure their paths inside `run_pipeline.py`.

## Running the Pipeline

Run this command from the root of the project:

```bash
py src/run_pipeline.py inputs/view0.png inputs/view1.png --grid 256 --image-size 256 --optimize-material
```

## Pipeline Stages

1. Reset output folders
2. Load binary silhouette images
3. Configure shadow sources and light directions
4. Compute conservative shadow hull
5. Simulate hull shadows and compute IoU metrics
6. Optimize silhouettes to reduce inconsistent shadow pixels
7. Recompute improved shadow hull
8. Hollow interior voxels while preserving outer shell
9. Export meshes
10. Save simulated shadow renders and debug slices

## Important Parameters

Inside `run_pipeline.py`:

- `nx, ny, nz` control voxel grid resolution
- `world_size` controls physical bounding box size
- `iterations` controls silhouette optimization strength
- `sample_per_view` controls optimization runtime vs quality
- `alpha` controls deformation step size
- `sigma` controls displacement smoothing strength
- `max_passes` controls carving aggressiveness
- shell thickness parameters control material optimization

Higher voxel resolution improves shadow accuracy but increases runtime and memory usage significantly.

## Output Files

The pipeline generates:

- `outputs/meshes/shadow_hull.stl`
- `outputs/meshes/shadow_carved.stl`
- Shadow simulation images per view
- Debug voxel slice images
- Console metrics including IoU, missing pixels, and reduction ratios

## Known Limitations

- Some silhouette combinations are geometrically incompatible
- Thin structures may disappear at low voxel resolution
- Hollowing may create disconnected internal artifacts without connectivity constraints
- Mesh surfaces are voxelized and may require smoothing
- Orthographic lighting model only
- Runtime increases quickly with grid resolution

## Future Improvements

- Connectivity preserving carving
- Signed distance shell generation
- Adaptive voxel grids
- GPU acceleration
- Mesh decimation and smoothing
- Support for more than two silhouettes
- Automatic light placement optimization
