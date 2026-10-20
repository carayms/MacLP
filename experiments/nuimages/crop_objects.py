"""Object crops from nuImages (Appendix J.2.1) and their label statistics.

For every annotated object of the selected city, the image content inside
the ground-truth bounding box is saved to

    <out-root>/<city>/<category>/<sample_token>_<i>.jpg

Ego-vehicle annotations and boxes smaller than 64 pixels in width or height
are skipped.  Run once for the training metadata (v1.0-train, Table S2) and
once for the validation metadata (v1.0-val, Table S3), each for both cities.
``--summary-only`` prints the per-class counts and proportions of an existing
crop folder.

Requires the nuImages devkit (``pip install nuscenes-devkit``) and the
nuImages images and metadata from https://www.nuscenes.org/nuimages.

Example:
    python nuimages/crop_objects.py --dataroot /data/nuimages --version v1.0-train \
        --out-root data/nuimages/crops_train
    python nuimages/crop_objects.py --out-root data/nuimages/crops_train --summary-only
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exputils.plotting import generate_stats_markdown_table

# The 15 categories used in the paper (classes with at least 100 crops); the
# order defines the labels 0..14.
SELECTED_CLASSES = [
    "human.pedestrian.adult", "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility", "movable_object.barrier",
    "movable_object.debris", "movable_object.pushable_pullable",
    "movable_object.trafficcone", "static_object.bicycle_rack",
    "vehicle.bicycle", "vehicle.bus.rigid", "vehicle.car",
    "vehicle.construction", "vehicle.motorcycle", "vehicle.trailer", "vehicle.truck",
]
CITIES = ["singapore", "boston"]


def tokens_by_city(dataset):
    """Sample tokens grouped by the city of their log."""
    tokens = {city: [] for city in CITIES}
    for sample in dataset.sample:
        location = dataset.get("log", sample["log_token"])["location"].lower()
        for city in CITIES:
            if city in location:
                tokens[city].append(sample["token"])
    return tokens


def annotations_by_sample(dataset):
    """Object annotations grouped by sample token (in the order of ``object_ann``)."""
    sample_of_sd = {sd["token"]: sd["sample_token"] for sd in dataset.sample_data}
    groups = defaultdict(list)
    for ann in dataset.object_ann:
        groups[sample_of_sd[ann["sample_data_token"]]].append(ann)
    return groups


def crop_city(dataset, tokens, out_root, city, min_size=64):
    """Save the object crops of one city; returns the number of crops written."""
    anns_of_sample = annotations_by_sample(dataset)
    n_written = 0
    for sample_token in tqdm(tokens, desc=f"Cropping {city}"):
        anns = anns_of_sample.get(sample_token, [])
        if not anns:
            continue
        img_info = dataset.get("sample_data", anns[0]["sample_data_token"])
        full_img = Image.open(os.path.join(dataset.dataroot, img_info["filename"])).convert("RGB")

        for i, ann in enumerate(anns):
            cat_name = dataset.get("category", ann["category_token"])["name"]
            if cat_name == "vehicle.ego":
                continue
            x1, y1, x2, y2 = ann["bbox"]
            if x2 - x1 < min_size or y2 - y1 < min_size:     # drop small, distant objects
                continue
            cat_dir = Path(out_root) / city / cat_name
            cat_dir.mkdir(parents=True, exist_ok=True)
            full_img.crop((x1, y1, x2, y2)).save(cat_dir / f"{sample_token}_{i}.jpg")
            n_written += 1
    return n_written


def count_crops(out_root):
    """counts[city][class] for the selected classes."""
    counts = {}
    for city in CITIES:
        city_dir = Path(out_root) / city
        if not city_dir.is_dir():
            raise FileNotFoundError(f"{city_dir} not found")
        counts[city] = np.array([len(list((city_dir / c).glob("*.jpg"))) if (city_dir / c).is_dir() else 0
                                 for c in SELECTED_CLASSES])
    return counts


def print_summary(out_root):
    """Counts, p(y), q(y) and w(y) = q(y) / p(y) as in Tables S2 and S3."""
    counts = count_crops(out_root)
    sg, bos = counts["singapore"], counts["boston"]
    p, q = sg / sg.sum(), bos / bos.sum()
    print(f"{'Category':<36} | {'Singapore':>9} | {'Boston':>7} | {'p(y)':>7} | {'q(y)':>9} | {'w(y)':>8}")
    print("-" * 92)
    for k, cat in enumerate(SELECTED_CLASSES):
        ratio = q[k] / p[k] if p[k] > 0 else float("nan")
        print(f"{cat:<36} | {sg[k]:>9,} | {bos[k]:>7,} | {p[k]:>7.4f} | {q[k]:>9.2e} | {ratio:>8.4f}")
    print("-" * 92)
    print(f"{'TOTAL':<36} | {sg.sum():>9,} | {bos.sum():>7,}")

    header = ["class"] + [f"{c.split('.')[-1]}.{k}" for k, c in enumerate(SELECTED_CLASSES)]
    with np.errstate(divide="ignore", invalid="ignore"):
        table = np.vstack([p, q, q / p])
    print("\n" + generate_stats_markdown_table(table, None, header, ["p(y) SG", "q(y) BOS", "omega"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataroot", help="nuImages root folder (images and metadata)")
    parser.add_argument("--version", default="v1.0-train", choices=["v1.0-train", "v1.0-val", "v1.0-mini"])
    parser.add_argument("--out-root", required=True, help="folder for the crops")
    parser.add_argument("--cities", nargs="+", default=CITIES, choices=CITIES)
    parser.add_argument("--min-size", type=int, default=64, help="minimum width and height of a box (pixels)")
    parser.add_argument("--summary-only", action="store_true", help="only print the statistics of --out-root")
    args = parser.parse_args()

    if not args.summary_only:
        if args.dataroot is None:
            parser.error("--dataroot is required unless --summary-only is given")
        from nuimages import NuImages

        nuim = NuImages(dataroot=args.dataroot, version=args.version, verbose=True)
        tokens = tokens_by_city(nuim)
        print(", ".join(f"{city}: {len(tokens[city])} images" for city in CITIES))
        for city in args.cities:
            n = crop_city(nuim, tokens[city], args.out_root, city, args.min_size)
            print(f"{city}: {n} crops written")
    print_summary(args.out_root)


if __name__ == "__main__":
    main()
