"""Repaint pixels in provinces.png whose colour is not listed in definition.csv.

CK3 turns every undefined colour into an extra "ghost" province with no title,
no locators and no region, which crashes the game once the daily/monthly tick
touches it. Each stray pixel is repainted with the most common defined colour
among its neighbours (falling back to the nearest defined colour).

Usage: python fix_province_colors.py path/to/map_data
Writes provinces_fixed.png next to the original; review it, then replace.
Requires Pillow and numpy (pip install pillow numpy).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_defined(path):
    defined = set()
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = line.split(";")
        if parts and parts[0].strip().isdigit() and len(parts) >= 4:
            defined.add(tuple(int(p) for p in parts[1:4]))
    return defined


def main(map_dir):
    map_dir = Path(map_dir)
    defined = load_defined(map_dir / "definition.csv")
    img = np.array(Image.open(map_dir / "provinces.png").convert("RGB"))
    h, w, _ = img.shape

    packed = (img[..., 0].astype(np.int32) << 16) | (img[..., 1].astype(np.int32) << 8) | img[..., 2]
    defined_packed = np.array([(r << 16) | (g << 8) | b for r, g, b in defined], dtype=np.int32)
    bad = ~np.isin(packed, defined_packed)
    ys, xs = np.nonzero(bad)
    if len(ys) == 0:
        print("No undefined colours found.")
        return

    for c, n in Counter(map(tuple, img[bad].tolist())).most_common():
        print(f"undefined colour RGB{c}: {n} px")

    defined_arr = np.array(sorted(defined), dtype=np.int32)
    todo = set(zip(ys.tolist(), xs.tolist()))
    while todo:
        fixed = {}
        for y, x in todo:
            votes = Counter()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (dy or dx) and 0 <= ny < h and 0 <= nx < w and not bad[ny, nx]:
                        votes[tuple(img[ny, nx])] += 1
            if votes:
                fixed[(y, x)] = votes.most_common(1)[0][0]
        if not fixed:  # isolated block with no defined neighbours
            for y, x in todo:
                d = ((defined_arr - img[y, x].astype(np.int32)) ** 2).sum(axis=1)
                fixed[(y, x)] = tuple(defined_arr[d.argmin()])
        for (y, x), c in fixed.items():
            img[y, x] = c
            bad[y, x] = False
        todo -= fixed.keys()
        print(f"repainted {len(fixed)} px ({len(todo)} left)")

    out = map_dir / "provinces_fixed.png"
    Image.fromarray(img).save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "map_data")
