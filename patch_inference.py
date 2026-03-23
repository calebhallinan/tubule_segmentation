import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Device selection: CUDA > MPS (Apple Silicon) > CPU
import torch as _torch
if _torch.cuda.is_available():
    DEVICE = "cuda"
elif _torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json
import argparse
import sys
from typing import List, Dict, Tuple
import warnings
from shapely.geometry import Polygon as ShapePolygon
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

# ============== Configuration ==============
IMG_SIZE = 1024
DINO_SIZE = 1022
OUTPUT_SIZE = 256
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DISTANCE_MODEL_PATH = "checkpoints/best_distance_model.pt"

ERODE_ITER = 5
MIN_AREA = 10             # minimum seed pixels in distance map space
THRESH_FACTOR = 0.5       # lower = more foreground included
MIN_CONFIDENCE = 0.6
MERGE_OVERLAP_THRESH = 0.3

# Seed detection via h-maxima (replaces peak_local_max distance/threshold tuning)
# SMOOTH_SIGMA: Gaussian blur applied before peak detection to kill noise ripples
# PEAK_H: a peak must rise at least this much above its surrounding basin to be a seed.
#   Low value  → more seeds (may split large tubules)
#   High value → fewer seeds (may miss small tubules)
SMOOTH_SIGMA = 2.0
PEAK_H = 0.03

# Tubule size filters (in pixels at full resolution)
MIN_TUBULE_AREA = 800     # lowered from 2000 to keep smaller tubules
MAX_TUBULE_AREA = 400000


# ============== Model Definitions ==============

class AlignmentBridge(nn.Module):
    """
    Fuses high-dimensional features from OpenMidnight (1536-dim) and SAM2 (256-dim)
    into a unified representation space for distance transform prediction.
    """
    def __init__(self, midnight_dim=1536, sam_dim=256, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(midnight_dim + sam_dim, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, out_dim, kernel_size=1)
        )
    
    def forward(self, m_feat, s_feat):
        m_up = F.interpolate(m_feat, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.proj(torch.cat([m_up, s_feat], dim=1))


class DistanceHead(nn.Module):
    """
    Decodes the fused features from the AlignmentBridge into a 
    single-channel predicted distance transform map using transpose convolutions.
    """
    def __init__(self, in_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 128, kernel_size=4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.head(x)


class DistanceModel(nn.Module):
    """
    End-to-end network combining the AlignmentBridge and DistanceHead to 
    predict distance maps from OpenMidnight and SAM2 feature embeddings.
    """
    def __init__(self):
        super().__init__()
        self.bridge = AlignmentBridge()
        self.head = DistanceHead()
    
    def forward(self, m_feat, s_feat):
        return self.head(self.bridge(m_feat, s_feat))


# ============== Core Functions ==============

def load_models():
    """Load all required models."""
    print("Loading models...")
    
    dl_loc = hf_hub_download(repo_id="SophontAI/OpenMidnight", filename="teacher_checkpoint_load.pt")
    midnight = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=False)
    cp = torch.load(dl_loc, map_location=DEVICE, weights_only=False)
    midnight.pos_embed = nn.Parameter(cp["pos_embed"])
    midnight.load_state_dict(cp)
    midnight = midnight.to(DEVICE).eval()
    print("  ✓ Midnight loaded")

    sam2_enc = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE).image_encoder.eval()
    sam2_predictor = SAM2ImagePredictor(build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE))
    print("  ✓ SAM2 loaded")

    model = DistanceModel().to(DEVICE).eval()
    model.load_state_dict(torch.load(DISTANCE_MODEL_PATH, map_location=DEVICE, weights_only=False)['model'])
    print("  ✓ Distance model loaded")
    
    return midnight, sam2_enc, sam2_predictor, model


@torch.inference_mode()
def extract_features(midnight, sam2_enc, img_tensor):
    """
    Extract feature embeddings from both OpenMidnight and SAM2 encoders.
    """
    img_midnight = F.interpolate(img_tensor, size=(DINO_SIZE, DINO_SIZE), mode='bilinear', align_corners=False)
    tokens = midnight.forward_features(img_midnight)["x_norm_patchtokens"]
    grid = int(tokens.shape[1] ** 0.5)
    m_feat = tokens.permute(0, 2, 1).reshape(1, 1536, grid, grid).contiguous()
    
    output = sam2_enc(img_tensor)
    s_feat = output["backbone_fpn"][-1]
    if s_feat.dim() == 3:
        if s_feat.shape[-1] == 256:
            s_feat = s_feat.permute(0, 2, 1)
        H = int(s_feat.shape[-1] ** 0.5)
        s_feat = s_feat.reshape(1, -1, H, H).contiguous()
    
    return m_feat, s_feat


def distance_to_instances(distance_map):
    """Convert distance map to instance masks using h-maxima + watershed.

    Key idea: smooth the distance map, then use h-maxima to find only peaks
    that rise meaningfully (>= PEAK_H) above their surrounding basin.
    Internal ripples within a single large tubule are tiny and get suppressed;
    the real valley between two distinct tubules is deep, so both peaks survive.
    This prevents large tubules from being split while still finding small ones.
    """
    # Binarize using Otsu threshold with leniency factor
    dist_uint8 = (distance_map * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_cv = cv2.threshold(dist_uint8, int(thresh_val * THRESH_FACTOR), 255, cv2.THRESH_BINARY)
    binary = (binary_cv > 0)

    # Smooth to suppress noise ripples before peak detection
    smoothed = gaussian_filter(distance_map.astype(np.float64), sigma=SMOOTH_SIGMA)

    # h-maxima: keep only peaks that are at least PEAK_H above their basin floor.
    # Restrict to foreground so background noise can't become seeds.
    peak_mask = h_maxima(smoothed, h=PEAK_H) & binary

    if not peak_mask.any():
        return []

    markers, _ = ndi.label(peak_mask)

    # Watershed on the smoothed map — consistent with peak detection
    labels = watershed(-smoothed, markers, mask=binary)

    masks = []
    for label_id in range(1, labels.max() + 1):
        mask = (labels == label_id).astype(np.uint8)
        if mask.sum() >= MIN_AREA:
            masks.append(mask)

    return masks


def refine_with_sam2(image, rough_masks, sam2_predictor, distance_map=None):
    """Refine rough masks using SAM2.

    Improvements over baseline:
    - Adds a positive point prompt at the peak of the distance map inside each
      instance, giving SAM2 a strong center-of-mass cue for better localization.
    - Filters out predictions with SAM2 confidence below MIN_CONFIDENCE to
      remove low-quality false positives.
    """
    if not rough_masks:
        return [], []

    sam2_predictor.set_image(image)
    refined, scores = [], []
    scale = IMG_SIZE / OUTPUT_SIZE

    for mask in rough_masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        # Format the mask hint for SAM2's expected input (-3 to +3 range)
        mask_hint = cv2.resize(mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_tensor = torch.from_numpy((mask_hint * 6.0) - 3.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Bounding box enclosing the rough instance
        box = np.array([xs.min() * scale, ys.min() * scale, xs.max() * scale, ys.max() * scale])

        # Point prompt at the distance-map peak inside this instance.
        # The highest-distance pixel is the tubule center — the strongest
        # possible localization hint for SAM2.
        point_coords, point_labels = None, None
        if distance_map is not None:
            peak_idx = distance_map[ys, xs].argmax()
            point_coords = np.array([[float(xs[peak_idx]) * scale, float(ys[peak_idx]) * scale]])
            point_labels = np.array([1])

        preds, score_preds, _ = sam2_predictor.predict(
            box=box,
            mask_input=mask_tensor,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        best = score_preds.argmax()
        best_score = float(score_preds[best])

        # Discard low-confidence predictions
        if best_score < MIN_CONFIDENCE:
            continue

        refined.append(preds[best])
        scores.append(best_score)

    return refined, scores


def save_geojson_safe(geojson_data: dict, output_path: str) -> bool:
    """
    Safely save GeoJSON with validation.
    Writes to temp file first, then moves to final location.
    """
    import tempfile
    import shutil
    
    output_path = Path(output_path)
    
    # Validate structure before saving
    print(f"   Validating {len(geojson_data['features'])} features...")
    
    valid_features = []
    for i, feature in enumerate(geojson_data['features']):
        try:
            coords = feature['geometry']['coordinates'][0]
            
            # Check each coordinate is [x, y] only (2D)
            valid_coords = []
            for pt in coords:
                if len(pt) >= 2:
                    # Force 2D - take only first two values
                    valid_coords.append([float(pt[0]), float(pt[1])])
                else:
                    print(f"   Warning: Feature {i} has invalid coordinate: {pt}")
                    continue
            
            # Ensure closed polygon
            if valid_coords and valid_coords[0] != valid_coords[-1]:
                valid_coords.append([valid_coords[0][0], valid_coords[0][1]])
            
            # Need at least 4 points
            if len(valid_coords) < 4:
                print(f"   Warning: Feature {i} has too few coordinates ({len(valid_coords)}), skipping")
                continue
            
            # Update feature with validated coordinates
            feature['geometry']['coordinates'] = [valid_coords]
            valid_features.append(feature)
            
        except Exception as e:
            print(f"   Warning: Feature {i} validation failed: {e}")
            continue
    
    print(f"   Valid features: {len(valid_features)} / {len(geojson_data['features'])}")
    
    geojson_data['features'] = valid_features
    
    # Write to temp file first
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as tmp:
            json.dump(geojson_data, tmp)
            tmp_path = tmp.name
        
        # Verify temp file is valid JSON
        with open(tmp_path, 'r') as f:
            _ = json.load(f)  # This will raise if invalid
        
        # Move to final location
        shutil.move(tmp_path, output_path)
        print(f"   ✓ Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to save: {e}")
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        return False


def masks_to_geojson_features(masks, scores, offset_x=0, offset_y=0, id_prefix="",
                               min_area=MIN_TUBULE_AREA, max_area=MAX_TUBULE_AREA):
    """Convert binary masks to QuPath-compatible GeoJSON features with validation."""
    features = []
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        try:
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            # 1. Preliminary area check and simplification
            # Simplification (epsilon=0.5) removes micro-artifacts from SAM2
            poly = cv2.approxPolyDP(contour, 0.5, True)
            if len(poly) < 3:
                continue
            
            # 2. Extract and Round coordinates (1 decimal place is plenty for WSI)
            points = poly.reshape(-1, 2)
            raw_coords = []
            for pt in points:
                x = round(float(pt[0] + offset_x), 1)
                y = round(float(pt[1] + offset_y), 1)
                raw_coords.append((x, y))
            
            if len(raw_coords) < 3:
                continue

            # 3. Use Shapely to fix "Reduction" / Topology errors
            # buffer(0) is a standard trick to fix self-intersecting polygons
            shape = ShapePolygon(raw_coords).buffer(0)
            
            if shape.is_empty or not shape.is_valid:
                continue

            # Handle potential MultiPolygons if buffer(0) split a self-intersector
            if shape.geom_type == 'MultiPolygon':
                shape = max(shape.geoms, key=lambda a: a.area)
            
            # Final area filter using validated geometry
            if shape.area < min_area or shape.area > max_area:
                continue

            # 4. Final GeoJSON Coordinate formatting
            # exterior.coords provides CCW winding order by default
            final_coords = [list(pt) for pt in shape.exterior.coords]

            features.append({
                "type": "Feature",
                "id": f"{id_prefix}{i}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [final_coords]
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": "Tubule",
                        "colorRGB": -65536  # QuPath Red
                    },
                    "isLocked": False,
                    "measurements": [
                        {"name": "Confidence", "value": round(float(score), 4)},
                        {"name": "Area px", "value": round(float(shape.area), 1)}
                    ]
                }
            })
            
        except Exception as e:
            # Silently skip truly broken geometries
            continue
    
    return features


def merge_overlapping_features(features, overlap_thresh=MERGE_OVERLAP_THRESH):
    """Merge GeoJSON features whose polygons overlap significantly.

    Two polygons are merged when their intersection area exceeds overlap_thresh
    times the area of the smaller polygon. Merging is iterated until stable so
    chains of overlapping instances collapse correctly. The merged feature keeps
    the highest confidence score of the group.
    """
    if len(features) <= 1:
        return features

    geoms = []
    confs = []
    for f in features:
        try:
            geoms.append(ShapePolygon(f['geometry']['coordinates'][0]))
            confs.append(f['properties']['measurements'][0]['value'])
        except Exception:
            geoms.append(None)
            confs.append(0.0)

    merged_into = list(range(len(geoms)))  # union-find: each starts as its own group

    def find(i):
        while merged_into[i] != i:
            merged_into[i] = merged_into[merged_into[i]]
            i = merged_into[i]
        return i

    for i in range(len(geoms)):
        if geoms[i] is None:
            continue
        for j in range(i + 1, len(geoms)):
            if geoms[j] is None:
                continue
            if find(i) == find(j):
                continue
            try:
                inter = geoms[i].intersection(geoms[j])
                if inter.is_empty:
                    continue
                ratio = inter.area / min(geoms[i].area, geoms[j].area)
                if ratio > overlap_thresh:
                    merged_into[find(j)] = find(i)
            except Exception:
                continue

    # Collect groups and union their geometries
    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(len(geoms)):
        if geoms[i] is not None:
            groups[find(i)].append(i)

    result = []
    for root, members in groups.items():
        union_geom = geoms[members[0]]
        best_conf = confs[members[0]]
        for idx in members[1:]:
            try:
                union_geom = union_geom.union(geoms[idx]).buffer(0)
                if union_geom.geom_type == 'MultiPolygon':
                    union_geom = max(union_geom.geoms, key=lambda g: g.area)
            except Exception:
                pass
            best_conf = max(best_conf, confs[idx])

        final_coords = [list(pt) for pt in union_geom.exterior.coords]
        result.append({
            "type": "Feature",
            "id": str(len(result)),
            "geometry": {"type": "Polygon", "coordinates": [final_coords]},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": "Tubule", "colorRGB": -65536},
                "isLocked": False,
                "measurements": [
                    {"name": "Confidence", "value": round(float(best_conf), 4)},
                    {"name": "Area px", "value": round(float(union_geom.area), 1)}
                ]
            }
        })

    return result


# ============== Main ==============

def make_versioned_output_dir(base_dir: Path) -> Path:
    """Return a unique output directory, auto-incrementing if base_dir already exists.

    results/HUK1_COR_1   → used if it doesn't exist
    results/HUK1_COR_1_2 → next attempt, and so on
    """
    if not base_dir.exists():
        base_dir.mkdir(parents=True)
        return base_dir
    n = 2
    while True:
        candidate = base_dir.parent / f"{base_dir.name}_{n}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        n += 1


def save_visualization(image: np.ndarray, features: list, out_path: Path):
    """Save a side-by-side visualization of the patch and its segmentations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image)
    axes[0].set_title("Original Patch", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].set_title(f"Segmented Tubules (n={len(features)})", fontsize=13)
    axes[1].axis("off")

    cmap = plt.cm.get_cmap("tab20", 20)
    for i, feat in enumerate(features):
        coords = np.array(feat["geometry"]["coordinates"][0])
        color = cmap(i % 20)
        axes[1].add_patch(plt.Polygon(coords, fill=True, alpha=0.35,
                                      facecolor=color, edgecolor=color, linewidth=1.5))
        conf = feat["properties"]["measurements"][0]["value"]
        axes[1].text(coords[:, 0].mean(), coords[:, 1].mean(), f"{conf:.2f}",
                     fontsize=6, ha="center", va="center",
                     color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


@torch.inference_mode()
def run(image_path, output_dir_base):
    """
    Main entry point for patch-level inference pipeline.

    Outputs are written to a versioned directory so previous runs are never
    overwritten.  If output_dir_base already exists the next available suffix
    is used (e.g. _2, _3, …).

    Saves per run:
      patch.png          – copy of the input image
      segmentation.geojson
      visualization.png  – side-by-side overlay
    """
    out_dir = make_versioned_output_dir(Path(output_dir_base))
    print(f"Output directory: {out_dir}")

    midnight, sam2_enc, sam2_predictor, model = load_models()

    print(f"Loading image: {image_path}")
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

    # Save a copy of the input patch into the run directory
    cv2.imwrite(str(out_dir / "patch.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 1. Resize image to model resolution
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) if (h != IMG_SIZE or w != IMG_SIZE) else image

    img_norm = (image_resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    # 2. Forward pass
    m_feat, s_feat = extract_features(midnight, sam2_enc, img_tensor)
    distance_map = model(m_feat, s_feat)[0, 0].cpu().numpy()

    # 3. Instance extraction and SAM2 refinement
    rough_masks = distance_to_instances(distance_map)
    refined_masks, scores = refine_with_sam2(image_resized, rough_masks, sam2_predictor, distance_map)

    # Scale masks back to original image size if needed
    if h != IMG_SIZE or w != IMG_SIZE:
        refined_masks = [
            cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            for m in refined_masks
        ]

    features = masks_to_geojson_features(refined_masks, scores, min_area=MIN_TUBULE_AREA, max_area=MAX_TUBULE_AREA)
    features = merge_overlapping_features(features)
    print(f"   After merging overlaps: {len(features)} tubules")

    # 4. Save outputs
    geojson_path = out_dir / "segmentation.geojson"
    save_geojson_safe({"type": "FeatureCollection", "features": features}, geojson_path)

    viz_path = out_dir / "visualization.png"
    save_visualization(image, features, viz_path)
    print(f"   ✓ Visualization: {viz_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python patch_inference.py <input_image> <output_dir>")
        print("  output_dir is auto-versioned: existing dirs get a _2, _3 … suffix")
        sys.exit(1)

    run(sys.argv[1], sys.argv[2])