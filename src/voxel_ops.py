import numpy as np

def make_voxel_centers(nx, ny, nz, world_size):
    """
    Input: # of voxels per axis and physical size of the voxel cube
    Returns: the 3D locations for every voxel center
    """

    # create the x, y, z positions for voxel centers
    xs = np.linspace(-0.5, 0.5, nx, endpoint=False) + 0.5 / nx
    ys = np.linspace(-0.5, 0.5, ny, endpoint=False) + 0.5 / ny
    zs = np.linspace(-0.5, 0.5, nz, endpoint=False) + 0.5 / nz

    # create and return the 3D coordinate grid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1) * world_size
    return pts  # [nx, ny, nz, 3]

def voxel_pitch(world_size, nx, ny, nz):
    # calculate the size of one voxel cube edge
    return world_size / max(nx, ny, nz)