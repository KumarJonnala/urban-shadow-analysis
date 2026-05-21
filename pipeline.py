"""Pipeline entry point.

Subcommands:
  download   — fetch orthophoto tile grid for all configured areas
  segment    — run OSM + vegetation segmentation on downloaded tiles
  all        — download then segment (default when no subcommand given)

Examples:
  python pipeline.py                     # download + segment
  python pipeline.py download            # tiles only
  python pipeline.py download --dry-run  # preview tile layout
  python pipeline.py segment             # segment only (tiles must exist)
  python pipeline.py all                 # explicit end-to-end
"""

import argparse

import numpy as np
from PIL import Image

from src.config import AREAS, OUTPUT_DIR, TILE_SIZE_M
from src.data_preprocessing import fetch_area_grid, tiles_for_area
from src.segmentation import fetch_buildings, fetch_roads, save_segmentation, vari_mask


def cmd_download(dry_run: bool = False) -> None:
    for area_name, area in AREAS.items():
        tiles = tiles_for_area(area, TILE_SIZE_M)
        print(f"\n--- {area_name}: {len(tiles)} tile(s) ---")
        if dry_run:
            for t in tiles:
                print(f"  tile ({t['ix']},{t['iy']})  W={t['west']:.6f} S={t['south']:.6f} E={t['east']:.6f} N={t['north']:.6f}")
        else:
            paths = fetch_area_grid(area_name)
            print(f"  {len(paths)} tile(s) saved")


def cmd_segment() -> None:
    seg_dir = OUTPUT_DIR / "segments"

    for area_name, area in AREAS.items():
        tiles = tiles_for_area(area, TILE_SIZE_M)
        tile_dir = OUTPUT_DIR / area_name
        print(f"\n--- {area_name}: segmenting {len(tiles)} tile(s) ---")

        buildings = fetch_buildings(
            area,
            cache_path=OUTPUT_DIR / f"buildings_{area_name}.geojson",
        )
        roads = fetch_roads(
            area,
            cache_path=OUTPUT_DIR / f"roads_{area_name}.geojson",
        )
        print(f"  OSM: {len(buildings)} buildings, {len(roads)} roads")

        for t in tiles:
            stem = f"{area_name}_tile_{t['ix']}_{t['iy']}"
            tile_path = tile_dir / f"{stem}.png"
            if not tile_path.exists():
                print(f"  [skip] {stem}.png not found — run 'download' first")
                continue

            img = np.array(Image.open(tile_path).convert("RGB"))
            tree_mask = vari_mask(img)
            npy_path, png_path = save_segmentation(
                img, t, buildings, roads, tree_mask,
                out_dir=seg_dir, stem=stem,
            )
            print(f"  {stem}: saved {npy_path.name}, {png_path.name}")


def cmd_all(dry_run: bool = False) -> None:
    cmd_download(dry_run=dry_run)
    if not dry_run:
        cmd_segment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urban shadow analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Fetch orthophoto tile grid")
    dl.add_argument("--dry-run", action="store_true", help="Show tile layout without downloading")

    sub.add_parser("segment", help="Segment downloaded tiles (OSM + vegetation)")

    all_cmd = sub.add_parser("all", help="Download then segment")
    all_cmd.add_argument("--dry-run", action="store_true", help="Show tile layout without downloading")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(dry_run=args.dry_run)
    elif args.command == "segment":
        cmd_segment()
    elif args.command == "all":
        cmd_all(dry_run=args.dry_run)
    else:
        cmd_all()
