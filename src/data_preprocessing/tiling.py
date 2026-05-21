"""Pure geometry: split a WGS84 bounding box into a regular grid of tiles."""

import math
from pyproj import Transformer

# EPSG:25832 (UTM 32N) used for metric tile size calculations
_to_m = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
_to_deg = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)


def tiles_for_area(area, tile_size_m):
    """Return list of tile dicts (west/south/east/north + grid ix/iy) for the given bbox."""
    min_x, min_y = _to_m.transform(area["west"], area["south"])
    max_x, max_y = _to_m.transform(area["east"], area["north"])
    nx = math.ceil((max_x - min_x) / tile_size_m)
    ny = math.ceil((max_y - min_y) / tile_size_m)
    tiles = []
    for ix in range(nx):
        for iy in range(ny):
            tmin_x = min_x + ix * tile_size_m
            tmin_y = min_y + iy * tile_size_m
            tmax_x = min(tmin_x + tile_size_m, max_x)
            tmax_y = min(tmin_y + tile_size_m, max_y)
            west, south = _to_deg.transform(tmin_x, tmin_y)
            east, north = _to_deg.transform(tmax_x, tmax_y)
            tiles.append({"west": west, "south": south, "east": east, "north": north, "ix": ix, "iy": iy})
    return tiles
