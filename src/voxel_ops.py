import numpy as np

def make_voxel_centers(nx, ny, nz, world_size):
    """
    Input: # of voxels per axis and physical size of the voxel cube
    Returns: the 3D locations for every voxel center
    """

    # create the x, y, z positions for voxel centers
    xs = (np.linspace(-0.5, 0.5, nx, endpoint=False, dtype=np.float32) + 0.5 / nx).astype(np.float32)
    ys = (np.linspace(-0.5, 0.5, ny, endpoint=False, dtype=np.float32) + 0.5 / ny).astype(np.float32)
    zs = (np.linspace(-0.5, 0.5, nz, endpoint=False, dtype=np.float32) + 0.5 / nz).astype(np.float32)

    # create and return the 3D coordinate grid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = (np.stack([X, Y, Z], axis=-1) * world_size).astype(np.float32)
    return pts  # [nx, ny, nz, 3]

def voxel_pitch(world_size, nx, ny, nz):
    # calculate the size of one voxel cube edge
    return world_size / max(nx, ny, nz)
