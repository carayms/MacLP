"""Resize all crops to 224 x 224 (bicubic) once before training (Appendix J.2.1).

The folder structure ``<city>/<category>/*.jpg`` is kept.

Example:
    python nuimages/resize_crops.py --src data/nuimages/crops_train --dst data/nuimages/crops_train_224
    python nuimages/resize_crops.py --src data/nuimages/crops_val --dst data/nuimages/crops_val_224
"""

import argparse
import os

from PIL import Image
from tqdm import tqdm


def resize_folder(src_root, dst_root, size=(224, 224)):
    images = [os.path.join(root, f) for root, _, files in os.walk(src_root)
              for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for path in tqdm(images, desc="Resizing"):
        target = os.path.join(dst_root, os.path.relpath(path, src_root))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with Image.open(path) as img:
            img.resize(size, Image.BICUBIC).save(target)
    return len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    n = resize_folder(args.src, args.dst, (args.size, args.size))
    print(f"{n} images resized to {args.size} x {args.size} in {args.dst}")


if __name__ == "__main__":
    main()
