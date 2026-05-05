# Multidirectional Light Sculptures

## Eric Nohara-LeClair, Yingtong (Sophie) Shen, Josh Ilano

The multidirectional light sculpture generation pipeline takes as input one to three reference silhouette images and generates a single component 3D printable voxel sculpture who's orthographic shadows match the inputs'. The pipeline creates the initial shadow hull, optimizes the input silhouettes to reduce inconsistencies among the input silhouettes, prunes redundant voxels, then hollows out the interior of the sculpture to reduce the materials used to print it.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Input Silhouettes](#input-silhouettes)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Web UI](#running-the-web-ui-streamlit-app)
- [Pipeline Stages](#pipeline-stages)
- [Important Parameters](#important-parameters)
- [Output Files](#output-files)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

## Features

- Shadow hull construction from multiple input silhouettes
- Support for custom multi-direction lighting / custom light directions
- Silhouette optimization to reduce missing shadow regions
- Voxel pruning for material optimization
- Hollow shell generation for material reduction
- Shadow simulation before fabrication
- Mesh export to STL format
- Debug visualizations including voxel slices and shadow comparisons
- Simple Streamlit UI and CLI

## Repository Structure

```bash
C:.
│   .gitignore
│   packages.txt
│   README.md
│   requirements.txt
│
├───assets
│       favicon.png
│
├───inputs
│       view0.png
│       view1.png
│       view2.png
│       view3.png
│       view4.png
│       view5.png
│       view6.png
│       view7.png
│       view8.png
│       view9.png
│
├───outputs
│   │
│   ├───run_2026-05-04_17-17-40
│   │   ├───debug
│   │   │   ├───masks
│   │   │   │   ├───base
│   │   │   │   └───opt
│   │   │   └───slices
│   │   ├───meshes
│   │   │       shadow_hull.stl
│   │   └───sim
│   └───sim
│
├───renders
│
└───src
    │   app.py
    │   carve.py
    │   debug_slices.py
    │   deform.py
    │   distances.py
    │   export_mesh.py
    │   image_io.py
    │   optimize.py
    │   optimize_consistency.py
    │   phylopic_api.py
    │   postprocess_prune.py
    │   projections.py
    │   render.py
    │   render_preview.py
    │   run_pipeline.py
    │   shadow_hull.py
    │   shadow_source.py
    │   simulate.py
    │   voxel_ops.py
    │   warp.py
    │
    └───__pycache__
```

## Requirements

We have listed our dependencies in our requirements.txt file. Please install them before attempting to run locally.

```bash
pip install -r requirements.txt
```

## Input Silhouettes

Input silhouettes are be binary images where:

- White pixels represent required shadow coverage
- Black pixels represent empty shadow

You may attempt to use with non binary images; however, the conversion from non binary images to binary images may lose a lot of the detail in the image. For best results, choose input images with clear dark and light regions. For best consistency, choose input images that are similar sizes/shapes.

Put silhouette images inside /inputs and input their paths on the command line when running the pipeline. If using the Streamlit UI, you may upload your own silhouette images or choose silhouettes from the integrated silhouette API.

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

## Running the Web UI (Streamlit App)

This project also includes an interactive Streamlit interface for generating sculptures without using the command line. To launch the app, run this command from the project root.

### Launch the app

```bash
streamlit run src/app.py
```

### Hosted app

Alternatively, we have hosted the streamlit app. To access, click on this link:

[Click to view hosted app](https://multidirectional-light-sculptures.streamlit.app/)

## Pipeline Stages

1. Load binary silhouette images as binary matrices
2. Configure shadow sources and light directions
3. Optimize silhouettes to reduce inconsistent shadow pixels between silhouettes
4. Compute conservative shadow hull
5. Prune redundant voxels and ensure single component constraint on structure
6. Hollow out the sculpture
7. Simulate hull shadows
8. Export carved mesh as STL
9. Save simulated shadow renders and debug slices to output run directory

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
- `directions` specifies custom lighting directions for each view

Higher voxel resolution improves shadow accuracy but increases runtime and memory usage. For best results with minal runtime, we reccomend using a grid and image resolution of 150.

## Output Files

The pipeline generates:

- `outputs/meshes/shadow_hull.stl`
- Shadow simulation images per view
- Debug silhouette masks
- Debug voxel slice images
- Metrics printed to the terminal

## Known Limitations

- Some silhouette combinations are geometrically incompatible and will lead to many missed pixels in the silhouette. No matter how they are optimized, some geometries just **dont work** together. The pipeline will attempt to generate a structure, but the quality of the output may be compromised
- Mesh surfaces are voxelized and may require smoothing if you desire a smooth sculpture (sanding or computationally)
- Orthographic lighting model only (assumes parallel light rays)
- Runtime increases heavily with grid + image resolution

## Future Improvements

- Mesh smoothing
- Automatic light placement optimization
- Some feedback to the user for their silhouettes **before** attempting to generate the structure (allow for better UX)
