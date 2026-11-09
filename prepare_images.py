"""Center the eight original ring photos on matching white square canvases."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image


SOURCES = {
    "sodium-reflection-upright.png": "f599a94ec41d595d8164e592d999cd4c.png",
    "sodium-reflection-inverted.png": "d16692f61eaba9006e13c5245e715621.png",
    "sodium-transmission-upright.png": "fd0609a5f45ffd57272b2ac90617cf5e.png",
    "sodium-transmission-inverted.png": "6da749c81909599a5f2ef84b202b3885.png",
    "white-reflection-upright.jpg": "e89e9dd4644af78df516996f1b0356cc.jpg",
    "white-reflection-inverted.png": "8c71025564b9efe4f210c8ae3cd48c00.png",
    "white-transmission-upright.png": "ac897a30884531435f11d3c068118f71.png",
    "white-transmission-inverted.png": "056aa1c73738cb20ae8424390635bf87.png",
}

SIZE = 960
RADIUS = 420


def prepare(source: Path, destination: Path) -> None:
    original = Image.open(source).convert("RGB")
    pixels = np.asarray(original)
    if source.suffix.lower() == ".jpg":
        foreground = pixels.min(axis=2) < 235
    else:
        foreground = pixels.max(axis=2) > 30

    ys, xs = np.nonzero(foreground)
    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    source_radius = max(right - left, bottom - top) / 2
    scale = source_radius / RADIUS

    result = original.transform(
        (SIZE, SIZE),
        Image.Transform.AFFINE,
        (scale, 0, center_x - scale * SIZE / 2,
         0, scale, center_y - scale * SIZE / 2),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white",
    )

    if source.suffix.lower() != ".jpg":
        yy, xx = np.ogrid[:SIZE, :SIZE]
        distance = np.hypot(xx - SIZE / 2, yy - SIZE / 2)
        alpha = np.clip(RADIUS + 0.5 - distance, 0, 1)[..., None]
        result_pixels = np.asarray(result).astype(np.float32)
        result = Image.fromarray(
            np.uint8(np.round(result_pixels * alpha + 255 * (1 - alpha)))
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".jpg":
        result.save(destination, quality=95, subsampling=0)
    else:
        result.save(destination, optimize=True)
    print(f"{destination.name}: center=({center_x:.1f}, {center_y:.1f}), radius={source_radius:.1f}")


if __name__ == "__main__":
    source_directory = Path(sys.argv[1])
    image_directory = Path(__file__).parent / "images"
    for destination_name, source_name in SOURCES.items():
        prepare(source_directory / source_name, image_directory / destination_name)
