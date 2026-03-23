"""
WSI Tubule Segmentation Script

Segments tubules from whole slide images using a trained distance model
with SAM2 refinement. Processes large images by tiling with batched inference.
"""

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
import tifffile
from PIL import Image
from typing import List, Dict, Tuple
import warnings
from shapely.geometry import Polygon as ShapePolygon
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
warnings.filterwarnings('ignore')

# ============== Configuration ==============
IMG_SIZE = 1024
DINO_SIZE = 1022
OUTPUT_SIZE = 256
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DISTANCE_MODEL_PATH = "checkpoints/best_distance_model.pt"

ERODE_ITER = 5
MIN_AREA = 30
THRESH_FACTOR = 0.8
MIN_CONFIDENCE = 0.7

# Tubule size filters (in pixels at full resolution)
MIN_TUBULE_AREA = 2000
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
    
    # Load Midnight (DINOv2)
    dl_loc = hf_hub_download(repo_id="SophontAI/OpenMidnight", filename="teacher_checkpoint_load.pt")
    midnight = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=False)
    cp = torch.load(dl_loc, map_location=DEVICE, weights_only=False)
    midnight.pos_embed = nn.Parameter(cp["pos_embed"])
    midnight.load_state_dict(cp)
    midnight = midnight.to(DEVICE).eval()
    print("  ✓ Midnight loaded")

    # Load SAM2 encoder and predictor
    sam2_enc = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE).image_encoder.eval()
    sam2_predictor = SAM2ImagePredictor(build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE))
    print("  ✓ SAM2 loaded")

    # Load distance model
    model = DistanceModel().to(DEVICE).eval()
    model.load_state_dict(torch.load(DISTANCE_MODEL_PATH, map_location=DEVICE, weights_only=False)['model'])
    print("  ✓ Distance model loaded")
    
    return midnight, sam2_enc, sam2_predictor, model


@torch.inference_mode()
def extract_features_batch(midnight, sam2_enc, img_tensors):
    """
    Extract features from Midnight and SAM2 encoders for a batch of images.
    
    Args:
        midnight: Midnight model
        sam2_enc: SAM2 encoder
        img_tensors: Tensor of shape (B, 3, H, W)
    
    Returns:
        m_feats: Midnight features (B, 1536, grid, grid)
        s_feats: SAM2 features (B, 256, H, W)
    """
    B = img_tensors.shape[0]
    
    # 1. Extract Midnight features (resizes input to 1022 for DINOv2 expected size)
    img_midnight = F.interpolate(img_tensors, size=(DINO_SIZE, DINO_SIZE), mode='bilinear', align_corners=False)
    tokens = midnight.forward_features(img_midnight)["x_norm_patchtokens"]
    grid = int(tokens.shape[1] ** 0.5)
    m_feats = tokens.permute(0, 2, 1).reshape(B, 1536, grid, grid).contiguous()
    
    # SAM2 features
    output = sam2_enc(img_tensors)
    s_feats = output["backbone_fpn"][-1]
    if s_feats.dim() == 3:
        if s_feats.shape[-1] == 256:
            s_feats = s_feats.permute(0, 2, 1)
        H = int(s_feats.shape[-1] ** 0.5)
        s_feats = s_feats.reshape(B, -1, H, H).contiguous()
    
    return m_feats, s_feats


def distance_to_instances(distance_map):
    """Convert distance map to instance masks using watershed segmentation.

    Uses local maxima of the distance map as seeds (tubule centers) and
    grows each region via watershed. This handles touching tubules more
    robustly than fixed erosion because it adapts to tubule size and shape.
    """
    # Binarize using Otsu threshold with leniency factor
    dist_uint8 = (distance_map * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(dist_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_cv = cv2.threshold(dist_uint8, int(thresh_val * THRESH_FACTOR), 255, cv2.THRESH_BINARY)
    binary = (binary_cv > 0)

    # Find local maxima in the distance map — these are the tubule centers.
    # min_distance=8 prevents two peaks being placed in the same tubule.
    # threshold_rel=0.3 ignores weak bumps below 30% of the image maximum.
    coords = peak_local_max(distance_map, min_distance=8, threshold_rel=0.3, labels=binary)

    if len(coords) == 0:
        return []

    # Build a marker image: each peak becomes a uniquely labelled seed
    peak_mask = np.zeros(distance_map.shape, dtype=bool)
    peak_mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = ndi.label(peak_mask)

    # Watershed grows each seed uphill through the distance map.
    # Negative sign because skimage watershed finds minima.
    labels = watershed(-distance_map, markers, mask=binary)

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


def save_geojson_streaming(features, output_path):
    """Writes GeoJSON features one-by-one to avoid memory exhaustion."""
    output_path = Path(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write the GeoJSON Header
        f.write('{"type":"FeatureCollection","features":[\n')
        
        for i, feat in enumerate(features):
            # Add a comma between features, but not before the first or after the last
            separator = ",\n" if i > 0 else ""
            f.write(separator + json.dumps(feat))
            
        # Write the GeoJSON Footer
        f.write('\n]}')
    
    print(f"   ✓ Successfully streamed {len(features)} features to {output_path}")

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
            # Silently skip truly broken geometries to keep the WSI processing moving
            continue
    
    return features


# ============== WSI Processor ==============

class WSITubuleSegmenter:
    """Segment tubules from whole slide images with batched inference."""
    
    def __init__(self, 
                 tile_size: int = 512,
                 overlap: int = 256,
                 batch_size: int = 4,
                 min_tissue_fraction: float = 0.1,
                 min_tubule_area: int = MIN_TUBULE_AREA,
                 max_tubule_area: int = MAX_TUBULE_AREA):
        """
        Initialize WSI segmenter.
        
        Args:
            tile_size: Size of tiles (must be 1024 for model)
            overlap: Overlap between tiles in pixels
            batch_size: Number of tiles to process in parallel
            min_tissue_fraction: Minimum fraction of tile that must contain tissue
            min_tubule_area: Minimum tubule area in pixels
            max_tubule_area: Maximum tubule area in pixels
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.min_tissue_fraction = min_tissue_fraction
        self.min_tubule_area = min_tubule_area
        self.max_tubule_area = max_tubule_area
        
        # Load models once
        self.midnight, self.sam2_enc, self.sam2_predictor, self.model = load_models()
    
    def load_wsi(self, image_path: str) -> Tuple[np.ndarray, int, int]:
        """Load WSI and return image with dimensions."""
        print(f"Loading WSI: {image_path}")
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        file_size_gb = path.stat().st_size / (1024**3)
        if file_size_gb > 5:
            print(f"  Warning: Large file ({file_size_gb:.1f} GB), this may take a while...")
        
        try:
            image = tifffile.imread(image_path)
        except Exception as e:
            print(f"  tifffile failed ({e}), trying PIL...")
            try:
                Image.MAX_IMAGE_PIXELS = None
                image = np.array(Image.open(image_path))
            except Exception as e2:
                raise RuntimeError(f"Failed to load image: {e2}")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] > 4:
                image = image[:, :, :3]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        if image.dtype != np.uint8:
            if image.max() > 255:
                image = (image / image.max() * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        h, w = image.shape[:2]
        print(f"  ✓ Image loaded: {w} x {h} pixels, {image.dtype}")
        
        return image, h, w
    
    def get_tile_info(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        """Get tile coordinates."""
        step = self.tile_size - self.overlap
        tiles = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                tiles.append((y, x, y_end, x_end))
        return tiles
    
    def is_tissue_tile(self, tile: np.ndarray) -> bool:
        """Check if tile contains enough tissue."""
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < 240
        tissue_fraction = tissue_mask.sum() / tissue_mask.size
        return tissue_fraction >= self.min_tissue_fraction
    
    def prepare_tile(self, tile: np.ndarray) -> np.ndarray:
        """Prepare a tile for model input (resize and normalize)."""
        h, w = tile.shape[:2]
        
        if h != IMG_SIZE or w != IMG_SIZE:
            tile = cv2.resize(tile, (IMG_SIZE, IMG_SIZE))
        
        img_norm = (tile.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return img_norm
    
    @torch.inference_mode()
    def process_batch(self, tiles: List[np.ndarray], 
                      original_tiles: List[np.ndarray]) -> List[Tuple[List[np.ndarray], List[float]]]:
        """
        Process a batch of tiles and return masks with scores.
        
        Args:
            tiles: List of prepared tiles (normalized)
            original_tiles: List of original tiles (for SAM2 refinement)
        
        Returns:
            List of (masks, scores) tuples for each tile
        """
        try:
            B = len(tiles)
            
            # Stack into batch tensor
            batch_tensor = torch.from_numpy(np.stack([
                t.transpose(2, 0, 1) for t in tiles
            ])).float().to(DEVICE)
            
            # Extract features for entire batch
            m_feats, s_feats = extract_features_batch(self.midnight, self.sam2_enc, batch_tensor)
            
            # Predict distance maps for entire batch
            distance_maps = self.model(m_feats, s_feats).cpu().numpy()
            
            # Process each tile's distance map individually (instance extraction + SAM2)
            results = []
            for i in range(B):
                distance_map = distance_maps[i, 0]
                
                # Convert to instances
                rough_masks = distance_to_instances(distance_map)

                # Refine with SAM2 (requires original image)
                orig_tile = original_tiles[i]
                if orig_tile.shape[0] != IMG_SIZE or orig_tile.shape[1] != IMG_SIZE:
                    orig_tile_resized = cv2.resize(orig_tile, (IMG_SIZE, IMG_SIZE))
                else:
                    orig_tile_resized = orig_tile

                refined_masks, scores = refine_with_sam2(orig_tile_resized, rough_masks, self.sam2_predictor, distance_map)
                
                # Scale masks back if needed
                h, w = original_tiles[i].shape[:2]
                if h != IMG_SIZE or w != IMG_SIZE:
                    scaled_masks = []
                    for mask in refined_masks:
                        scaled = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        scaled_masks.append(scaled.astype(bool))
                    refined_masks = scaled_masks
                
                results.append((refined_masks, scores))
            
            return results
            
        except Exception as e:
            print(f"\n  Warning: Failed to process batch: {e}")
            return [([], []) for _ in tiles]
    
    def process_wsi(self, 
                    image_path: str, 
                    output_dir: str,
                    visualize_tiles: int = 5) -> List[Dict]:
        """
        Process entire WSI and save results.
        
        Args:
            image_path: Path to WSI
            output_dir: Output directory
            visualize_tiles: Number of tiles to visualize (0 to disable)
            
        Returns:
            List of all GeoJSON features
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load WSI
        image, h, w = self.load_wsi(image_path)
        
        # Get tiles
        tiles_info = self.get_tile_info(h, w)
        print(f"Total tiles: {len(tiles_info)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tubule size filter: {self.min_tubule_area} - {self.max_tubule_area} px²")
        
        # Filter to tissue tiles first
        tissue_tiles = []
        for tile_idx, (y, x, y_end, x_end) in enumerate(tiles_info):
            tile = image[y:y_end, x:x_end].copy()
            
            # Pad if needed
            if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                padded = np.ones((self.tile_size, self.tile_size, 3), dtype=np.uint8) * 255
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            if self.is_tissue_tile(tile):
                tissue_tiles.append((tile_idx, y, x, tile))
        
        print(f"Tissue tiles: {len(tissue_tiles)} / {len(tiles_info)}")
        
        all_features = []
        visualized = 0
        
        # Process in batches
        for batch_start in range(0, len(tissue_tiles), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(tissue_tiles))
            batch_info = tissue_tiles[batch_start:batch_end]
            
            # Prepare batch
            prepared_tiles = []
            original_tiles = []
            for _, _, _, tile in batch_info:
                prepared_tiles.append(self.prepare_tile(tile))
                original_tiles.append(tile)
            
            # Process batch
            batch_results = self.process_batch(prepared_tiles, original_tiles)
            
            # Collect results
            for i, (tile_idx, y, x, tile) in enumerate(batch_info):
                masks, scores = batch_results[i]
                
                if masks:
                    features = masks_to_geojson_features(
                        masks, scores,
                        offset_x=x, offset_y=y,
                        id_prefix=f"tile{tile_idx:04d}_",
                        min_area=self.min_tubule_area,
                        max_area=self.max_tubule_area
                    )
                    all_features.extend(features)
                    
                    # Visualize some tiles
                    if visualized < visualize_tiles and len(masks) > 0:
                        self._visualize_tile(tile, masks, scores,
                                            output_path / f"tile_{tile_idx:04d}.png")
                        visualized += 1
            
            # Progress update
            tiles_done = min(batch_end, len(tissue_tiles))
            print(f"\rProcessed {tiles_done}/{len(tissue_tiles)} tissue tiles, "
                  f"tubules found: {len(all_features)}", end="")
            
            # Clear CUDA cache periodically
            if batch_end % (self.batch_size * 10) == 0 and DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        print(f"\n\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Tissue tiles processed: {len(tissue_tiles)}")
        print(f"Background tiles skipped: {len(tiles_info) - len(tissue_tiles)}")
        print(f"Total tubules found: {len(all_features)}")
        
        # Save GeoJSON using the streaming method
        geojson_path = output_path / "tubules.geojson"
        save_geojson_streaming(all_features, geojson_path)
        
        return all_features
    
    def process_single_tile(self,
                           image_path: str,
                           tile_idx: int,
                           output_dir: str) -> List[Dict]:
        """Process a single tile for debugging."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        image, h, w = self.load_wsi(image_path)
        tiles_info = self.get_tile_info(h, w)
        
        if tile_idx >= len(tiles_info):
            raise ValueError(f"Tile index {tile_idx} out of range (max: {len(tiles_info) - 1})")
        
        y, x, y_end, x_end = tiles_info[tile_idx]
        print(f"Processing tile {tile_idx}: [{y}:{y_end}, {x}:{x_end}]")
        
        tile = image[y:y_end, x:x_end].copy()
        if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
            padded = np.ones((self.tile_size, self.tile_size, 3), dtype=np.uint8) * 255
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        
        # Process as single-item batch
        prepared = [self.prepare_tile(tile)]
        results = self.process_batch(prepared, [tile])
        masks, scores = results[0]
        
        print(f"Found {len(masks)} tubules (before filtering)")
        
        features = masks_to_geojson_features(
            masks, scores,
            offset_x=x, offset_y=y,
            id_prefix=f"tile{tile_idx:04d}_",
            min_area=self.min_tubule_area,
            max_area=self.max_tubule_area
        )
        print(f"Kept {len(features)} tubules (after size filtering)")
        
        geojson_path = output_path / f"tile_{tile_idx:04d}.geojson"
        with open(geojson_path, 'w') as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
        
        self._visualize_tile(tile, masks, scores, output_path / f"tile_{tile_idx:04d}.png")
        
        return features
    
    def _visualize_tile(self, tile: np.ndarray, masks: List[np.ndarray],
                        scores: List[float], output_path: Path):
        """Create visualization of tile segmentation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        H, W = tile.shape[:2]

        norm_masks: List[np.ndarray] = []
        for m in masks:
            m = np.asarray(m)
            if m.dtype != np.bool_:
                m = m > 0
            if m.shape[:2] != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            norm_masks.append(m)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(tile)
        axes[0].set_title("Original Tile")
        axes[0].axis("off")

        axes[1].imshow(tile)
        axes[1].set_title(f"Segmented ({len(norm_masks)} tubules)")
        axes[1].axis("off")

        if len(norm_masks) == 0:
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return

        cmap = plt.cm.get_cmap("tab20", max(20, len(norm_masks)))

        for i, (m, score) in enumerate(zip(norm_masks, scores)):
            color = cmap(i % cmap.N)
            overlay = np.zeros((H, W, 4), dtype=np.float32)
            overlay[m] = (color[0], color[1], color[2], 0.40)
            axes[1].imshow(overlay)

            m_u8 = m.astype(np.uint8) * 255
            contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if len(c) < 3:
                    continue
                pts = c.reshape(-1, 2)
                axes[1].plot(pts[:, 0], pts[:, 1], linewidth=1.2)

            ys, xs = np.where(m)
            if xs.size:
                cx, cy = float(xs.mean()), float(ys.mean())
                s = float(score) if score is not None else float("nan")
                axes[1].text(
                    cx, cy, f"{i}:{s:.2f}",
                    fontsize=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Segment tubules from WSI')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input WSI path')
    parser.add_argument('--output-dir', '-o', type=str, default='wsi_tubule_output',
                       help='Output directory')
    parser.add_argument('--tile-idx', type=int, default=None,
                       help='Process single tile (for debugging)')
    parser.add_argument('--tile-size', type=int, default=512,
                       help='Tile size (default: 512)')
    parser.add_argument('--overlap', type=int, default=256,
                       help='Tile overlap (default: 256)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference (default: 8)')
    parser.add_argument('--min-tissue', type=float, default=0.1,
                       help='Minimum tissue fraction (default: 0.1)')
    parser.add_argument('--min-tubule-area', type=int, default=MIN_TUBULE_AREA,
                       help=f'Minimum tubule area in pixels (default: {MIN_TUBULE_AREA})')
    parser.add_argument('--max-tubule-area', type=int, default=MAX_TUBULE_AREA,
                       help=f'Maximum tubule area in pixels (default: {MAX_TUBULE_AREA})')
    parser.add_argument('--visualize', type=int, default=0,
                       help='Number of tiles to visualize (default: 5)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if args.min_tubule_area >= args.max_tubule_area:
        print(f"Error: min-tubule-area ({args.min_tubule_area}) must be less than max-tubule-area ({args.max_tubule_area})")
        return 1
    
    if args.batch_size < 1:
        print(f"Error: batch-size must be at least 1")
        return 1
    
    # Create segmenter
    segmenter = WSITubuleSegmenter(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        min_tissue_fraction=args.min_tissue,
        min_tubule_area=args.min_tubule_area,
        max_tubule_area=args.max_tubule_area
    )
    
    # Process
    if args.tile_idx is not None:
        features = segmenter.process_single_tile(args.input, args.tile_idx, args.output_dir)
        print(f"Processed tile {args.tile_idx}: {len(features)} tubules")
    else:
        features = segmenter.process_wsi(args.input, args.output_dir, 
                                         visualize_tiles=args.visualize)
        print(f"Total tubules: {len(features)}")
    
    return 0


if __name__ == "__main__":
    exit(main())