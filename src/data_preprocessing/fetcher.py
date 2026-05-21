"""WMS client: fetch a single orthophoto tile from the Sachsen-Anhalt DOP20 service."""

import requests
from pyproj import Transformer

from src.config import WMS_URL, WMS_LAYER, IMAGE_WIDTH, IMAGE_HEIGHT, OUTPUT_DIR

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def fetch_and_save(bbox: dict, filename: str, output_dir=OUTPUT_DIR, width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT):
    """Request a WMS GetMap for bbox (WGS84) and save the PNG to output_dir/filename."""
    min_x, min_y = _transformer.transform(bbox["west"], bbox["south"])
    max_x, max_y = _transformer.transform(bbox["east"], bbox["north"])
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "CRS": "EPSG:25832",
        "BBOX": f"{min_x},{min_y},{max_x},{max_y}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "TRANSPARENT": "false",
    }
    print(f"Fetching {filename} ...")
    r = requests.get(WMS_URL, params=params, timeout=60)
    r.raise_for_status()
    out_path = output_dir / filename
    out_path.write_bytes(r.content)
    print(f"  Saved {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path
