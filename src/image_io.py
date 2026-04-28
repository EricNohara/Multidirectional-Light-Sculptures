from PIL import Image
import numpy as np
import os
from scipy.ndimage import binary_closing, binary_opening, binary_dilation

def load_binary_image(
    path: str,
    size=(128, 128),
    threshold=128,
    invert=False,
    do_cleanup=True,
    close_iters=0,
    open_iters=0,
    dilate_iters=0,
    fill_holes=False,
):
    """
    Load image -> binary silhouette mask.
    Dark pixels become True by default.
    """

    img = Image.open(path).convert("RGBA").resize(size)

    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("L")

    arr = np.array(img)
    mask = arr < threshold

    if invert:
        mask = ~mask

    if do_cleanup:
        if close_iters > 0:
            for _ in range(close_iters):
                mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
        if open_iters > 0:
            for _ in range(open_iters):
                mask = binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        if dilate_iters > 0:
            for _ in range(dilate_iters):
                mask = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))

    return mask.astype(bool)

def save_mask(mask, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img.save(path)