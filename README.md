# Multidirectional Light Sculptures

## Eric Nohara-LeClair, Yingtong (Sophie) Shen, Josh

The multidirectional light sculpture generation pipeline takes as input one or more reference silhouette images and generates a 3D printable voxel sculpture who's orthographic shadows match the inputs'. The pipeline creates the initial shadow hull, optimizes the input silhouettes to reduce inconsistencies to reduce missing parts of the silhouettes, then hollows out the interior of the sculpture to reduce the materials used to print it.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Input Silhouettes](#input-silhouettes)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Stages](#pipeline-stages)
- [Important Parameters](#important-parameters)
- [Output Files](#output-files)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

## Features

- Shadow hull construction from multiple input silhouettes
- Support for custom multi-direction lighting / custom light directions
- Optional silhouette optimization to reduce missing shadow regions
- Priority-based hollow carving using per-pixel support counts
- Hollow shell generation for material reduction
- Shadow simulation before fabrication
- Mesh export to STL format for both raw and carved outputs
- Debug visualizations including voxel slices and shadow comparisons
- Progress bars for long running optimization and carving stages

## Repository Structure

```bash
C:.
│   README.md
│
├───inputs
│       views.png
│
├───renders
│       renders.png
│
├───outputs
│   ├───debug
│   │   ├───masks
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

Input silhouettes must be binary images where:

- White pixels represent required shadow coverage
- Black pixels represent empty shadow

Put silhouette images inside /inputs and input their paths on the command line when running the pipeline.

## Running the Pipeline

Run this command from the root of the project:

```bash
py src/run_pipeline.py inputs/view0.png inputs/view1.png --grid 256 --image-size 256 --optimize-material
```

Example with custom lighting directions:

```bash
python src/run_pipeline.py inputs/view4.png inputs/view6.png --grid 96 --image-size 128 --optimize-material --directions "1,0,0;1,0,1"
```
The --grid, --image-size, --optimize-material, and --directions flags are optional.
You must provide at least 2 silhouette images.
Custom directions should be given as semicolon-separated 3D vectors, one per input image.

## Pipeline Stages

1. Reset output folders
2. Load binary silhouette images as binary matrices
3. Configure shadow sources and light directions
4. Optionally optimize silhouettes to reduce inconsistent shadow pixels between silhouettes
5. Compute conservative shadow hull
6. Simulate hull shadows
7. Export raw hull mesh as STL
8. Hollow out the sculpture if optimizing materials
9. Simulate carved shadows
10. Export carved mesh as STL
11. Save simulated shadow renders and debug slices to output

## Important Parameters

Inside `run_pipeline.py`:

- `nx, ny, nz` control voxel grid resolution
- `world_size` controls physical bounding box size
- `iterations` controls silhouette optimization strength
- `sample_per_view` controls optimization runtime vs quality
- `alpha` controls deformation step size
- `sigma` controls displacement smoothing strength
- `max_passes` controls carving aggressiveness
- `shell_thickness_voxels` controls material optimization thickness
- `threshold` controls image binarization
- `close_iters`, `open_iters`, and `dilate_iters` control silhouette preprocessing
- `directions` specifies custom lighting directions for each view

Higher voxel resolution improves shadow accuracy but increases runtime and memory usage.

## Output Files

The pipeline generates:

- `outputs/meshes/shadow_hull.stl`
- `outputs/meshes/shadow_carved.stl`
- Shadow simulation images per view
- Debug silhouette masks
- Debug voxel slice images
- Metrics printed to the terminal

## Known Limitations

- Some silhouette combinations are geometrically incompatible and will lead to many missed pixels in the silhouette
- Thin structures may disappear at low voxel resolution
- Hollowing doesn't seem to affect print time or material
- Mesh surfaces are voxelized and require smoothing (sanding or computationally)
- Orthographic lighting model only (assumes parallel light rays)
- Runtime increases with grid resolution

## Future Improvements

- Better connectivity preserving carving
- Automatically adaptive voxel grids
- GPU acceleration
- Mesh smoothing
- Automatic light placement optimization
- Better preservation of small internal silhouette holes and fine details
