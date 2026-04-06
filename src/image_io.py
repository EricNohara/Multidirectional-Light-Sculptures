from PIL import Image
import numpy as np
import os

def load_binary_image(
    path: str,
    size=(128, 128),
    threshold=128,
    invert=False,
    pad_ratio=0.08,
) -> np.ndarray:
    """
    Load an image and convert it to a centered binary silhouette mask.

    Steps:
    - remove transparency
    - threshold to binary
    - crop to the silhouette bounding box
    - pad onto a square canvas
    - resize to target size

    This makes different input silhouettes aligned consistently even if the
    original images were cropped differently.
    """

    # load original image
    img = Image.open(path).convert("RGBA")

    # flatten transparency onto white
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("L")

    # grayscale -> binary mask
    arr = np.array(img)
    mask = arr < threshold
    if invert:
        mask = ~mask

    # if mask is empty, just return blank resized mask
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(size, dtype=bool)

    # tight bounding box of the silhouette
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = mask[y0:y1, x0:x1]

    h, w = cropped.shape

    # square canvas with a little padding
    side = max(h, w)
    pad = max(1, int(round(side * pad_ratio)))
    canvas_side = side + 2 * pad

    canvas = np.zeros((canvas_side, canvas_side), dtype=bool)

    # center the cropped silhouette on the square canvas
    y_off = (canvas_side - h) // 2
    x_off = (canvas_side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = cropped

    # resize to final output size
    out = Image.fromarray((canvas.astype(np.uint8) * 255))
    out = out.resize(size, Image.NEAREST)

    return (np.array(out) > 0)

def save_mask(mask, path):
    """
    Function to convert a boolean matrix back into an image. Used for debugging and testing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img.save(path)