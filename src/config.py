from dataclasses import dataclass
import numpy as np

@dataclass
class ShadowSource:
    image: np.ndarray          # boolean silhouette matrix [H, W], True = shadow/filled pixel
    direction: np.ndarray      # unit vector, orthographic light/view direction
    up: np.ndarray             # unit vector defining image vertical axis
    world_center: np.ndarray   # center of volume
    world_size: float          # cube side length in world units