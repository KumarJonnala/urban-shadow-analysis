# Pipeline Methods, Formulas, and Assumptions

Complete reference for all mathematical formulas, empirical constants, thresholds, and
modelling assumptions used in the urban-shadow-analysis pipeline.
Organised in order of pipeline execution.

---

## 1. Coordinate Reference Systems

All geographic bounding boxes (`AREAS` dict) are stored in **WGS84 (EPSG:4326)**.
All metric calculations — pixel-size derivation, crown areas, shadow lengths, spatial joins —
operate in **EPSG:25832 (UTM Zone 32N)**. Conversion is done once per tile with:

```python
Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
```

`src/shadow/casting.py:18`, `src/data_preprocessing/tiling.py:7–8`

---

## 2. Data Acquisition

### 2.1 Tile Grid Layout

An area bounding box is subdivided into a regular grid of equal tiles:

```
nx = ceil((east_m  - west_m)  / tile_size_m)
ny = ceil((north_m - south_m) / tile_size_m)
```

Boundary tiles are clipped to the area extent (not padded).
`src/data_preprocessing/tiling.py:15–16`

### 2.2 WMS Image Dimensions

Every tile is fetched from the WMS at a fixed pixel resolution:

| Constant | Value |
|---|---|
| `IMAGE_WIDTH` | 1200 px |
| `IMAGE_HEIGHT` | 1200 px |

`src/config.py:9–11`

### 2.3 Pixel Size per Tile Size

```
pixel_size_m = ((east_m - west_m) / W + (north_m - south_m) / H) / 2
```

Averages horizontal and vertical scales; treated as uniform across the tile.
`src/shadow/casting.py:23–28`

Derived effective resolutions for standard tile sizes (1200 × 1200 px):

| Tile size | pixel_size_m | Relation to DOP20 (0.2 m/px) |
|---|---|---|
| 100 m | ≈ 0.083 m/px | oversampled (sub-DOP20) |
| 250 m | ≈ 0.208 m/px | ≈ DOP20 native |
| 500 m | ≈ 0.417 m/px | undersampled ×2 |
| 1000 m | ≈ 0.833 m/px | undersampled ×4 |

### 2.4 Rasterio Affine Transform

Maps pixel (row, col) to EPSG:25832 metric coordinates using rasterio's
`from_bounds(west_m, south_m, east_m, north_m, W, H)`. Convention: pixel (0, 0) is
the NW corner; x (column) increases eastward, y (row) increases southward.
`src/shadow/casting.py:31–37`

---

## 3. Vegetation Segmentation

### 3.1 Segmentation Layer Priority Stacking

When tree, road, and building masks overlap, the following priority applies:

```
seg[building_mask] = 3   # lowest priority
seg[road_mask]     = 2
seg[tree_mask]     = 1   # highest priority — overwrites road/building
```

`src/segmentation/overlay.py:84–87`

### 3.2 VARI (Visible Atmospherically Resistant Index)

**Formula:**

```
VARI = (G − R) / (G + R − B + ε),   ε = 1e-6
```

**Threshold:** pixels with VARI > 0.05 → vegetation.

**Post-processing:**
- Morphological disk closing with radius 4 px (fills small canopy gaps)
- Remove connected components ≤ 500 px ≈ 20 m² (at DOP20 resolution)

`src/segmentation/vegetation.py:22–57`

### 3.3 DeepForest Aerial Tree Detector

| Parameter | Value | Note |
|---|---|---|
| `score_threshold` | 0.3 | keeps ~half of detections |
| `patch_size` | 400 px | sliding window size |
| `patch_overlap` | 0.05 | fractional overlap between patches |
| `iou_threshold` | 0.15 | NMS IoU for cross-window deduplication |
| `min_size` | 500 px | same noise filter as VARI |
| `closing_radius` | 4 px | same gap closing as VARI |

`src/segmentation/vegetation.py:73–80`

### 3.4 SAM + VARI Ensemble

SAM segment boundaries partition the image; segments are classified as vegetation if:
- Mean VARI within segment > 0.05
- Segment size ≥ 200 px

`src/segmentation/vegetation.py:229–234`

### 3.5 SegFormer-B5 (ADE20K)

ADE20K class indices treated as vegetation: `(4, 9, 17)` = tree, grass, plant.
`src/segmentation/vegetation.py:139`

### 3.6 TCD SegFormer

Model: `restor/tcd-segformer-mit-b5`. The model is scale-sensitive (trained on ~10 cm/px
global aerial imagery), so input images are resized before inference:

```
scale = 1024 / max(orig_h, orig_w)
new_h = int(orig_h × scale)
new_w = int(orig_w × scale)
```

Output is upsampled back to the original size with bilinear interpolation.
Binary classes: 0 = background, 1 = tree (`_TCD_TREE_CLASS = 1`).
`src/segmentation/vegetation.py:299–343`

### 3.7 DeepLabV3 ResNet50 (ImageNet pre-training)

Pre-processing normalisation:
```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```
Uses VOC class 16 (`pottedplant`) as the only available vegetation proxy.
**Note:** Performance on aerial orthophotos is expected to be poor — included for comparison only.
`src/segmentation/vegetation.py:453–454`

---

## 4. Tree Polygon Extraction

### 4.1 Minimum Component Size (Noise Filter)

Connected components with fewer than **50 pixels** are discarded.
At DOP20 resolution (0.2 m/px), 50 px ≈ 2 m².
`src/shadow/casting.py:55, 106, 137, 185`

### 4.2 Crown Area from Pixel Count

```
CPA_m² = n_pixels × pixel_size_m²
```

`src/shadow/casting.py:109, 187`

### 4.3 Crown Radius from Crown Area

Equivalent circular radius assuming a circular crown footprint:

```
r_m = sqrt(CPA_m² / π)
```

`src/shadow/casting.py:110, 141, 188`

### 4.4 Watershed Split — Global Threshold

Components whose equivalent crown radius exceeds `MAX_CROWN_RADIUS_M` are watershed-split
into individual crowns before height estimation. This prevents the √N height inflation that
occurs when N merged trees are treated as one giant tree.

```
MAX_CROWN_RADIUS_M = 8.0 m   (≈ 200 m² crown area)
```

`src/config.py:28`, `src/shadow/casting.py:112`

### 4.5 Watershed Split — Per-Genus Thresholds

Species-specific 95th-percentile crown radius replaces the global 8.0 m threshold when the
tile's dominant BK genus is known. Using p95 ensures single large crowns still pass unsplit
while true multi-tree clusters are split.

Selected values (full table in `src/config.py:109–167`):

| Genus | p95 radius | Median radius | n (BK 2026) |
|---|---|---|---|
| Quercus rubra | 11.0 m | 4.0 m | 479 |
| Salix alba | 10.0 m | 5.0 m | 671 |
| Populus nigra | 10.0 m | 5.0 m | 686 |
| Platanus acerifolia | 8.5 m | 4.0 m | 1510 |
| Tilia cordata | 6.5 m | 3.5 m | 10444 |
| Acer campestre | 6.0 m | 3.0 m | 3925 |
| Sorbus aria | 3.0 m | 2.0 m | 274 |

Fallback: `MAX_CROWN_RADIUS_M = 8.0 m` when genus is absent or BK clip is empty.
`pipeline.py:222`, `src/config.py:109–167`

### 4.6 Watershed — Minimum Peak Distance

Local maxima in the distance-transform EDT used to seed watershed markers must be separated by:

```
min_dist_px = max(1, int(max_crown_radius_m × 0.5 / pixel_size_m))
```

The 0.5× factor prevents over-segmentation of a single dense crown.
`src/shadow/casting.py:101`

### 4.7 Watershed — Single-Peak Fallback (Area Cap)

When only one peak is found in an oversized cluster (one big tree, not a cluster), the
component is not split. Height is estimated using the crown area capped to the maximum
single-crown area to prevent runaway estimates:

```
capped_radius = min(r_m, max_crown_radius_m)
capped_area   = π × capped_radius²
H_capped      = exp(A + B × ln(capped_area))
```

`src/shadow/casting.py:122–124`

### 4.8 Sub-Component Crown Radius Cap

After watershed splitting, each sub-component's crown radius attribute is capped:

```
sub_radius_m = min(sqrt(sub_area_m² / π), max_crown_radius_m)
```

The uncapped `sub_area_m²` still feeds the allometric formula (line 143).
`src/shadow/casting.py:141, 143`

### 4.9 Rendering Dissolve Snap Buffer

For visual rendering only (does not modify on-disk FGB data), a snap buffer closes
sub-pixel gaps between adjacent polygons:

```
snap_m = 0.3 m   (just above the 0.2 m/px DOP20 resolution)
result = union(buffer(polygons, +0.3 m)).buffer(−0.3 m)
```

`pipeline.py:84–100`

---

## 5. Allometric Height Model

### 5.1 Formula

Power-law allometric model fitted as OLS in log-log space:

```
H_m = exp(A + B × ln(CPA_m²))
```

Equivalently: `H = e^A × CPA^B`

where `CPA_m²` is the crown projected area (crown footprint) in m².

### 5.2 Global Coefficients

Fitted by OLS (`ln(H) ~ ln(CPA)`) on the full Magdeburg Baumkataster 2026
(`Baeume_SFM_2026.gpkg`), all species, citywide:

| Parameter | Value | Note |
|---|---|---|
| `ALLOMETRIC_A` | 1.317 | ln-space intercept |
| `ALLOMETRIC_B` | 0.318 | power-law exponent |
| R² | 0.53 | |
| n | 84,081 | street trees, valid measurements only |

Predictions match cadastre medians within ±10% for crown diameters 4–12 m.
Source: `references/Baeume_SFM_2026.gpkg`; fitted in `test_notebooks/baumkataster_analysis.ipynb`.
`src/config.py:34–40`, `src/shadow/casting.py:113, 143, 189`

### 5.3 Per-Genus Allometric Profiles

Same formula with genus-specific (A, B) coefficients. Fitted by OLS on species with
n ≥ 200 trees in the Baumkataster 2026. Falls back to global coefficients if the tile's
dominant genus is absent.

`ALLOMETRIC_PROFILES` in `src/config.py:45–103` — 56 species:

| Species | A | B | R² | n |
|---|---|---|---|---|
| Tilia cordata | 1.0799 | 0.3864 | 0.665 | 10,435 |
| Acer platanoides | 1.5002 | 0.2716 | 0.470 | 8,119 |
| Robinia pseudoacacia | 1.8419 | 0.2185 | 0.310 | 5,779 |
| Fraxinus excelsior | 1.4301 | 0.3106 | 0.500 | 5,607 |
| Quercus robur | 1.3798 | 0.3120 | 0.572 | 4,858 |
| Acer pseudoplatanus | 1.5826 | 0.2594 | 0.419 | 4,031 |
| Acer campestre | 1.4289 | 0.2706 | 0.483 | 3,924 |
| Aesculus hippocastanum | 1.3895 | 0.3096 | 0.569 | 2,479 |
| Carpinus betulus | 1.3151 | 0.3183 | 0.592 | 2,196 |
| Prunus avium | 1.3153 | 0.2522 | 0.428 | 1,750 |
| Acer negundo | 1.5969 | 0.2079 | 0.327 | 1,736 |
| Platanus acerifolia | 1.1471 | 0.3503 | 0.652 | 1,510 |
| Pyrus communis | 1.1895 | 0.2476 | 0.558 | 1,225 |
| Tilia platyphyllos | 1.5648 | 0.2706 | 0.460 | 1,209 |
| Populus canadensis Hybride | 2.2410 | 0.2012 | 0.290 | 1,109 |
| Tilia cordata 'Greenspire' | 1.4216 | 0.2484 | 0.687 | 863 |
| Ulmus laevis | 1.6273 | 0.2622 | 0.508 | 849 |
| Tilia euchlora | 1.2496 | 0.3605 | 0.602 | 838 |
| Populus nigra 'Italica' | 1.7121 | 0.4710 | 0.658 | 761 |
| Malus spec. | 1.0390 | 0.1942 | 0.357 | 757 |
| Pinus sylvestris | 2.4931 | 0.1042 | 0.082 | 737 |
| Prunus padus | 1.4955 | 0.1688 | 0.365 | 694 |
| Populus nigra | 1.9586 | 0.2363 | 0.515 | 686 |
| Salix alba | 1.3447 | 0.2721 | 0.402 | 670 |
| Acer platanoides 'Columnare' | 1.4811 | 0.2896 | 0.625 | 667 |
| Carpinus betulus 'Fastigiata' | 1.4099 | 0.2381 | 0.424 | 629 |
| Pinus nigra | 1.8269 | 0.3129 | 0.466 | 628 |
| Populus canadensis | 1.8341 | 0.2743 | 0.556 | 591 |
| Betula pendula | 1.6957 | 0.2660 | 0.424 | 589 |
| Crataegus monogyna | 1.4232 | 0.1304 | 0.135 | 481 |
| Quercus rubra | 1.1000 | 0.3689 | 0.691 | 479 |
| Juglans regia | 1.1396 | 0.2878 | 0.514 | 456 |
| Ailanthus altissima | 1.3760 | 0.3031 | 0.574 | 420 |
| Corylus colurna | 1.2245 | 0.3268 | 0.567 | 415 |
| Prunus spec. | 1.2495 | 0.2267 | 0.431 | 374 |
| Prunus serrulata 'Kanzan' | 1.3360 | 0.1607 | 0.441 | 331 |
| Styphnolobium japonicum | 1.1277 | 0.2794 | 0.769 | 322 |
| Malus sylvestris (communis) | 0.9744 | 0.2230 | 0.346 | 313 |
| Alnus glutinosa | 1.5591 | 0.2703 | 0.460 | 302 |
| Populus spec. | 1.7247 | 0.2763 | 0.597 | 300 |
| Prunus mahaleb | 1.1690 | 0.2582 | 0.402 | 280 |
| Sorbus aria | 1.0874 | 0.3165 | 0.463 | 274 |
| Aesculus carnea | 0.9382 | 0.3378 | 0.577 | 272 |
| Ulmus glabra | 1.4804 | 0.2898 | 0.633 | 263 |
| Gleditsia triacanthos | 1.3762 | 0.3001 | 0.688 | 256 |
| Salix spec. | 1.5703 | 0.2418 | 0.378 | 249 |
| Fraxinus ornus | 1.2656 | 0.2218 | 0.554 | 242 |
| Acer platanoides 'Globosum' | 0.8476 | 0.2581 | 0.601 | 241 |
| Pyrus calleryana 'Chanticleer' | 1.3303 | 0.2133 | 0.417 | 241 |
| Populus canescens | 1.6423 | 0.3165 | 0.527 | 238 |
| Tilia tomentosa | 1.1331 | 0.3086 | 0.752 | 238 |
| Tilia spec. | 1.3210 | 0.3260 | 0.643 | 237 |
| Sorbus aucuparia | 1.2865 | 0.2072 | 0.497 | 219 |
| Sorbus intermedia | 1.1311 | 0.2875 | 0.529 | 219 |
| Quercus robur 'Fastigiata' | 1.7241 | 0.3339 | 0.758 | 218 |
| Liquidambar styraciflua | 1.3298 | 0.2521 | 0.803 | 215 |
| Ulmus carpinifolia | 1.5622 | 0.2665 | 0.382 | 206 |

---

## 6. Baumkataster Enrichment

Source: `references/Baeume_SFM_2026.gpkg` — 85,302 Magdeburg street and public trees (2026).

### 6.1 BK Validity Filters

Only BK records with credible measurements are used for matching:

```
Baumhoehe         > 1.0 m    (height)
Kronendurchmesser > 0.5 m    (crown diameter)
```

Zero-valued `Stammumfang` (trunk circumference) and `Pflanzjahr` (planting year) are stored
as `None` rather than 0.
`src/shadow/cadastre.py:54–57, 110–115`

### 6.2 Candidate Buffer for Spatial Join

Each BK point is buffered to create a circular search zone:

```
buffer_radius = max(Kronendurchmesser / 2, 2.0 m)
```

The 2.0 m minimum ensures even the smallest registered trees have a finite search zone.
`src/shadow/cadastre.py:61–64`

### 6.3 Match Radius Threshold

After the spatial intersection candidate test, the nearest BK tree must lie within:

```
match_radius_m = 15.0 m
```

Centroid-to-point distance (pipeline polygon centroid → BK GPS point).
Accounts for positional error in both datasets and segmentation localisation uncertainty.
`src/shadow/cadastre.py:27, 88–89`

### 6.4 Tie-Breaking

When multiple BK trees intersect one pipeline polygon, the nearest by centroid-to-point
distance is selected:

```
sort by _dist ascending → groupby tree_id → keep first
```

`src/shadow/cadastre.py:87–93`

### 6.5 Height and Crown Radius Override

For matched trees, allometric estimates are replaced by measured BK values:

```
height_m       = Baumhoehe              (measured height in metres)
crown_radius_m = Kronendurchmesser / 2  (measured diameter / 2)
height_source  = "measured"
```

Unmatched trees retain allometric estimates with `height_source = "allometric"`.
The original allometric estimate is always stored in `allometric_height_m` for validation.
`src/shadow/cadastre.py:102–105`

### 6.6 Dominant Genus per Tile

The most frequent genus (first word of `Gattung lang`) among BK trees in the tile clip:

```python
bk_tile["Gattung lang"].str.split().str[0].value_counts().index[0]
```

Used to select per-genus allometric (A, B) and watershed split threshold.
`src/shadow/cadastre.py:19`, `pipeline.py:221`

### 6.7 Deciduousness Classification

A tree is classified as deciduous if its genus (first token of `Gattung lang`) appears in:

```python
DECIDUOUS_GENERA = {
    "Tilia", "Acer", "Quercus", "Fraxinus", "Robinia", "Aesculus",
    "Carpinus", "Prunus", "Platanus", "Populus", "Ulmus", "Betula",
    "Salix", "Fagus", "Sorbus", "Pyrus", "Malus",
}
```

Conifers (Pinus, Picea, etc.) are implicitly evergreen (`is_deciduous = False`).
`src/shadow/cadastre.py:8–12, 107`

---

## 7. Shadow Casting

### 7.1 Solar Position

Sun azimuth and elevation for a given UTC datetime are computed using **pysolar** at the
tile's geographic centre:

```
lat = (bbox["south"] + bbox["north"]) / 2
lon = (bbox["west"]  + bbox["east"])  / 2
```

This flat-tile approximation introduces < 0.01° azimuth/elevation error for 250 m tiles
at 52°N latitude.
`src/shadow/solar.py:9–11, 36–40`

**Azimuth convention:** `azimuth_deg = get_azimuth(...) % 360°`
(0° = North, clockwise positive — compass bearing).

### 7.2 Minimum Sun Elevation (Shadow Floor)

If solar elevation < **5.0°** (including nighttime negatives), no shadows are cast:

```
if elevation_deg < min_elevation_deg:   return all-False mask
```

`src/shadow/casting.py:258, 294–295`

### 7.3 Shadow Azimuth (Anti-Solar Direction)

```
shadow_az_rad = radians((azimuth_deg + 180°) % 360°)
```

`src/shadow/casting.py:298`

### 7.4 Shadow Length

```
shadow_length_m = min(H / tan(elevation_rad),  max_shadow_factor × crown_radius_m)
```

The cap (`max_shadow_factor = 5.0`) prevents runaway shadow lengths at very low sun angles.
`src/shadow/casting.py:321–322`

### 7.5 Shadow Pixel Offsets

```
dx_px =  shadow_length_m × sin(shadow_az_rad) / pixel_size_m   # column offset (east = +)
dy_px = −shadow_length_m × cos(shadow_az_rad) / pixel_size_m   # row offset (north = row 0)
```

The negative sign on `dy_px` converts geographic north (positive) to image rows (positive
downward, since row 0 is the north edge).
`src/shadow/casting.py:326–327`

### 7.6 Shadow Ray Stepping (Chebyshev Rasterisation)

The shadow corridor between crown and shadow tip is filled by stepping through intermediate
positions:

```
n_steps = max(|dy_i|, |dx_i|, 1)
for step in 0..n_steps:
    shift crown_mask by (dy_px × step/n_steps, dx_px × step/n_steps)
    accumulate into shadow_mask
```

One step per Chebyshev distance unit ensures no pixel gaps in the shadow corridor.
`src/shadow/casting.py:332–339`

### 7.7 Building Occlusion

Shadow propagation stops at the first step where the shifted crown mask overlaps any
building pixel:

```
if (shifted_crown & building_mask).any():   break
```

Assumption: buildings are fully opaque (no partial transmission). The entire crown's shadow
terminates when any part of it hits a building — conservative behaviour.
`src/shadow/casting.py:340–341`

### 7.8 Shadow Mask Post-Processing

Two classes are excluded from the final shadow mask:

```
shadow_mask &= ~(seg_map == 1)   # exclude tree-canopy pixels
shadow_mask &= ~(seg_map == 3)   # exclude building pixels
```

`src/shadow/casting.py:343–344`

---

## 8. Validation Metrics

### 8.1 Height Validation (Allometric vs. BK Measured)

Computed only for trees where `height_source == "measured"` (BK-matched).

```
error = allometric_height_m − height_m
bias  = mean(error)
MAE   = mean(|error|)
RMSE  = sqrt(mean(error²))
R²    = 1 − SS_res / SS_tot
```

Printed automatically at the end of `cmd_segment` after each tile-size merge.
`pipeline.py:250–258`

### 8.2 Multi-Size Pairwise IoU

Tree masks from different tile sizes are rasterised onto a common 1 m/px UTM grid and
compared pairwise:

```
IoU = intersection_pixels / union_pixels   (1.0 if union == 0)
```

`pipeline.py:384–387`

### 8.3 Precision / Recall / F1 vs. Reference Size

Each tile size is evaluated against a reference (default: 250 m):

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

where TP / FP / FN are pixel counts on the 1 m/px common grid.
`pipeline.py:401–407`

### 8.4 Tile-Summary Spatial Metrics

Computed per tile in `_save_tile_summary()` (`pipeline.py`) after each segment run and
saved to `outputs/segments/{size}m/tile_summary_{size}m.png`.

**BK match rate**

```
match_rate = n_matched / n_segmented × 100  (%)
```

`n_matched` = pipeline polygons with `height_source == "measured"` whose centroid falls in
the tile. `n_segmented` = all pipeline polygons in the tile. Low match rate indicates sparse
BK coverage (parks, private gardens, campus areas not registered as public street trees).

**Crown area ratio**

```
ratio = Σ crown_area_m²  (pipeline, all polygons in tile)
      / Σ π × (Kronendurchmesser / 2)²  (BK trees clipped to tile)
```

Ratio > 1: pipeline detects more canopy than BK registers (expected in park/campus tiles
where private trees and unmapped vegetation are present). Ratio < 1: pipeline under-segments
or BK contains large-crown trees not visible in the orthophoto.

**Mean tree height per tile**

```
mean_h = mean(height_m)   all pipeline polygons in tile
```

Uses the final `height_m` value (BK-measured if matched, allometric otherwise).
Useful for identifying tall-tree zones (river banks, old parks) vs. young street-tree corridors.

`pipeline.py:_save_tile_summary`

---

## 9. Key Constants Summary

| Constant | Value | Source file | Purpose |
|---|---|---|---|
| `ALLOMETRIC_A` | 1.317 | `config.py:39` | ln-space intercept, global allometry |
| `ALLOMETRIC_B` | 0.318 | `config.py:40` | power-law exponent, global allometry |
| `MAX_CROWN_RADIUS_M` | 8.0 m | `config.py:28` | watershed split threshold (global fallback) |
| `min_component_pixels` | 50 px | `casting.py:55` | noise filter for connected components |
| `min_elevation_deg` | 5.0° | `casting.py:258` | shadow floor — below this no shadow cast |
| `max_shadow_factor` | 5.0 | `casting.py:259` | shadow length cap (× crown_radius_m) |
| `match_radius_m` (BK) | 15.0 m | `cadastre.py:27` | max centroid-to-BK-point distance |
| BK buffer minimum | 2.0 m | `cadastre.py:64` | minimum candidate search radius |
| BK valid: height | > 1 m | `cadastre.py:56` | BK data quality filter |
| BK valid: crown | > 0.5 m | `cadastre.py:57` | BK data quality filter |
| `VARI threshold` | 0.05 | `vegetation.py:28` | spectral vegetation classifier |
| `VARI min_size` | 500 px ≈ 20 m² | `vegetation.py:28` | morphological noise removal |
| `VARI closing_radius` | 4 px | `vegetation.py:28` | morphological gap closing |
| `DeepForest score_threshold` | 0.3 | `vegetation.py:76` | detection confidence cutoff |
| `DeepForest patch_size` | 400 px | `vegetation.py:77` | sliding window size |
| `DeepForest patch_overlap` | 0.05 | `vegetation.py:78` | patch overlap fraction |
| `DeepForest iou_threshold` | 0.15 | `vegetation.py:79` | cross-window NMS threshold |
| `SAM min_segment` | 200 px | `vegetation.py:233` | SAM segment noise filter |
| `SAM VARI threshold` | 0.05 | `vegetation.py:232` | VARI gate for SAM segments |
| `TCD resize_to` | 1024 px | `vegetation.py:303` | scale normalisation for TCD SegFormer |
| `snap_m` (render) | 0.3 m | `pipeline.py:84` | dissolve buffer for rendering |
| `IMAGE_WIDTH` / `IMAGE_HEIGHT` | 1200 px | `config.py:9–11` | WMS request dimensions |
| `TILE_SIZE_M` (default) | 250 m | `config.py:16` | default tile edge length |
| `TILE_SIZES_M` | [100, 250, 500, 1000] | `config.py:19` | all tile sizes for multi-scale analysis |
| ε (VARI denominator) | 1e-6 | `vegetation.py:25` | numerical stability guard |
| min_dist_px factor | 0.5 | `casting.py:101` | watershed peak separation = 0.5 × max_r / px_size |
| Diurnal hour range | 04:00–20:00 UTC | `pipeline.py:641` | hourly shadow evaluation window |
| `BK n_min` (allometry) | 200 trees | notebook | minimum per-species sample for OLS fit |
| ImageNet mean (R, G, B) | 0.485, 0.456, 0.406 | `vegetation.py:453` | DeepLabV3 pre-processing |
| ImageNet std (R, G, B) | 0.229, 0.224, 0.225 | `vegetation.py:454` | DeepLabV3 pre-processing |
| SegFormer veg. classes | (4, 9, 17) | `vegetation.py:139` | ADE20K tree/grass/plant indices |
| TCD tree class | 1 | `vegetation.py:303` | binary class index for TCD model |
| compare-sizes resolution | 1.0 m/px | `pipeline.py:316` | common grid for multi-size IoU |
| compare-sizes reference | 250 m | `pipeline.py:319` | reference tile size for precision/recall |

---

## 10. Validation Outputs

The pipeline saves three validation artefacts automatically after each `segment` run.

### 10.1 Height Comparison (`_save_height_comparison`)

**What it measures:** how much the per-species allometric prediction deviates from the global
model across every segmented tree (not just BK-matched ones).

```
diff_pct = (allometric_height_m − h_global_m) / h_global_m × 100   [%]
```

**Output files:**
- `{area}_height_comparison_{size}m.png` — per-tile boxplots + overall histogram (display clipped to p2–p98)
- `{area}_height_comparison_{size}m.json` — per-tile and overall statistics

**Interpretation:** tiles with `mean_pct > +10%` are dominated by species with a high A-coefficient
(e.g. Populus canadensis, A=2.24) — their per-species model predicts taller trees than the global
baseline for the same crown area. Tiles with `mean_pct < −10%` are dominated by low-A species
(e.g. Tilia cordata, A=1.08), predicted shorter than the global model. Tiles near 0% contain species
whose per-species fit is close to the global coefficients. The overall distribution describes how
much the species composition of the area shifts estimated heights relative to a uniform city-wide
baseline.

`pipeline.py:_save_height_comparison`

### 10.2 Watershed Comparison (`_save_watershed_comparison`)

**What it measures:** the effect of watershed splitting on per-tile tree count and total crown area,
run before and after the watershed step to confirm correct behaviour.

**3-panel figure:**
- Panel 1: grouped bar chart of pre/post tree count per tile
- Panel 2: per-tile count ratio (post/pre); ratio > 1 indicates blobs were successfully split
- Panel 3: per-tile total crown area (pre/post bars) with area ratio on a secondary axis

**Key property — area conservation:** `area_ratio ≈ 1.000` across all tiles. Watershed splitting
redistributes pixels between sub-polygons without removing them; the only source of ratio < 1 is
the minimum-component filter (50 px ≈ 2 m²) discarding sub-threshold fragments after splitting.
In practice total area loss is < 5 m² per tile.

**Count ratio interpretation:** a high ratio in a tile with dense canopy or row plantings
confirms the watershed is decomposing multi-tree blobs into individual crowns. A ratio near 1.0
indicates either that no blobs exceeded the split threshold or that the tile had sparse tree cover.

`pipeline.py:_save_watershed_comparison`

### 10.3 Tile Summary PCT (`_save_tile_summary_pct`)

**What it measures:** the same per-tile statistics as `_save_tile_summary` but normalised and
displayed as a spatial heatmap grid (one cell per tile) rather than absolute counts.

**6-panel heatmap:** tree count, BK match rate (%), crown area ratio (pipeline/BK), mean tree
height, allometric R², and a spare panel for future metrics. The grid layout mirrors the actual
tile (ix, iy) positions, making spatial patterns immediately visible — e.g. low match-rate
tiles at the campus interior vs. high match-rate tiles along registered street corridors.

`pipeline.py:_save_tile_summary_pct`

---

## 11. Results — ovgu_bbox2 @ 250 m Tiles

### 11.1 Dataset

| Metric | Value |
|---|---|
| Area | ovgu_bbox2 (OVGU campus + surrounding streets, Magdeburg) |
| Tile grid | 9 × 9 = 81 tiles at 250 m |
| Total segmented trees | 4,560 |
| BK-matched trees | 1,088 (23.9 %) |
| BK source | Baeume_SFM_2026.gpkg |

Top matched species: Tilia cordata (211), Acer platanoides (178), Platanus acerifolia (78),
Tilia euchlora (51), Acer pseudoplatanus (40).

The 23.9 % match rate reflects the BK dataset covering only registered public street trees.
Campus interior courtyards, parks, private-property gardens, and unmapped green spaces are
absent from the BK — pipeline polygons in these areas have no BK counterpart and carry only
allometric height estimates.

### 11.2 Height Estimation Accuracy vs. BK Measured Heights

Evaluated on the 1,088 BK-matched trees using
`dev = (H_pred − H_BK) / H_BK × 100 [%]`:

| Model | Mean dev | Median dev | Std | \|dev\| > 20 % | \|dev\| > 50 % |
|---|---|---|---|---|---|
| Global (A=1.317, B=0.318) | +42.5 % | +20.5 % | 92.5 % | 62.9 % | 29.2 % |
| Per-species allometric | +39.7 % | +19.6 % | 87.3 % | 62.8 % | 27.8 % |

The positive mean bias indicates systematic overestimation. The dominant cause is **merged
crown blobs**: 18.2 % of matched trees (198 / 1,088) have pipeline `crown_area_m2 > 200 m²`,
exceeding the maximum plausible single-tree crown footprint. These segments represent multiple
trees whose canopies merged in the segmentation and were not fully decomposed by the watershed
splitter. For this subset, the mean height overestimation is **+134.6 %** (median +89.6 %).

Excluding merged blobs (CPA < 200 m², n = 890), the global model overestimation drops to
**mean +22.0 %, median +13.7 %** — substantially better, but still positive-biased due to
(a) BK heights being rounded to the nearest metre and (b) the allometric model's inherent
scatter (R² = 0.53 citywide).

The per-species refinement provides only marginal improvement (−2.8 pp mean, −0.9 pp median)
because the dominant species on the OVGU campus — Tilia and Acer — have allometric coefficients
close to the global fit (Tilia cordata A = 1.08 vs. global 1.317, but B = 0.39 vs. 0.318,
partially offsetting).

### 11.3 Per-Species vs. Global Model Comparison (4,102 trees)

Source: `ovgu_bbox2_height_comparison_250m.json`

| Metric | Value |
|---|---|
| Overall mean diff_pct | +2.35 % |
| Overall std | 17.1 % |
| Trees with \|diff\| > 5 % | 53.2 % |
| Trees with \|diff\| > 10 % | 28.6 % |

Most tiles fall within ±10 %. Tiles with strong species signal:

| Tile | Mean diff_pct | Likely cause |
|---|---|---|
| (7,4) | +69.9 % | Populus-dominated (A ≈ 1.93) |
| (8,4) | +60.8 % | Same |
| (5,8) | +24.0 % | High-CPA trees, strong A-coefficient species |
| (1,8) | +20.2 % | Mixed high-A species |
| (6,7) | −37.0 % | Tilia-dominated corridor (A = 1.08) |
| (4,1) | −19.0 % | Tilia / low-A species row planting |

### 11.4 Watershed Splitting Effect

Across all tiles in ovgu_bbox (confirmed from `ovgu_bbox_watershed_comparison_250m.png`):
- Crown area ratio post/pre: **≈ 1.000** — pixel-area fully conserved
- Tree count increases after splitting (merged blobs → individual crowns)
- Tiles with high-density linear plantings show the largest count ratios,
  confirming successful decomposition of row-planting clusters

### 11.5 Discussion

**Primary accuracy limitation — unsplit merged blobs:**
The allometric formula `H = exp(A + B·ln(CPA))` is monotonically increasing in CPA. When
N individual crowns merge into one polygon with area N × CPA_single, the estimated height
becomes `exp(A + B·ln(N·CPA_single)) ≈ N^B × H_single`. With B ≈ 0.32:

| Merged trees (N) | Height inflation factor |
|---|---|
| 2 | 2^0.32 ≈ 1.25× |
| 4 | 4^0.32 ≈ 1.55× |
| 16 | 16^0.32 ≈ 2.42× |

Watershed splitting is therefore the critical quality gate for height accuracy. The 8.0 m
maximum crown radius threshold (`MAX_CROWN_RADIUS_M`) catches most multi-tree blobs, but
dense grove patches and large ornamental trees near the threshold remain unsplit.

**BK match rate:** 23.9 % is not a pipeline failure. The BK registers only public street trees
maintained by the city; the unregistered 76.1 % are real trees (private property, campus
interiors, parks) for which the allometric model is the only height source.

**Allometric model fitness:** R² = 0.53 citywide is the ceiling for a single power-law model
fitted on all species and crown sizes. Per-species refinement brings R² above 0.65 for the
most common broadleaf genera (Tilia 0.665, Quercus rubra 0.691, Styphnolobium 0.769). The
model is least reliable for conifers (Pinus sylvestris R² = 0.082), where crown shape departs
fundamentally from the circular broadleaf assumption used for CPA estimation.
