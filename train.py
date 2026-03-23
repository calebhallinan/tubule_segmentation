import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ============== Configuration ==============
IMG_SIZE = 1024
DINO_SIZE = 1022
LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 30
OUTPUT_SIZE = 256
MAX_DISTANCE = 50
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
VISUALIZE_EVERY = 10


# ============== Distance Transform ==============

def compute_distance_transform(mask, max_distance=MAX_DISTANCE):
    """
    Compute normalized distance transform from binary mask.
    """
    if mask.max() > 1:
        binary = (mask > 127).astype(np.uint8)
    else:
        binary = (mask > 0.5).astype(np.uint8)
    
    if binary.sum() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    
    distance = ndimage.distance_transform_edt(binary).astype(np.float32)
    distance = np.clip(distance, 0, max_distance) / max_distance
    
    return distance


# ============== Visualization ==============

def visualize_batch(images, masks, targets, predictions, epoch, batch_idx, save_dir="distance_outputs"):
    """Visualize: Input | Mask | GT Distance | Prediction | Difference"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images_denorm = (images * std + mean).clamp(0, 1)
    
    num_samples = min(4, images.shape[0])
    fig, axes = plt.subplots(num_samples, 5, figsize=(17.5, 3.5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    titles = ['Input', 'GT Mask', 'GT Distance', 'Prediction', 'Difference']
    
    for i in range(num_samples):
        # Get data
        img = images_denorm[i].cpu().permute(1, 2, 0).numpy()
        mask = masks[i, 0].cpu().numpy()
        gt = targets[i, 0].cpu().numpy()
        pred = predictions[i, 0].detach().cpu().numpy()
        
        # Get sizes
        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape
        
        # Col 0: Input image (original size)
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        
        # Col 1: GT mask overlay - RESIZE mask to match image
        mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask_resized, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        axes[i, 1].axis('off')
        
        # Col 2: GT distance (at OUTPUT_SIZE)
        axes[i, 2].imshow(gt, cmap='hot', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        
        # Col 3: Predicted distance (at OUTPUT_SIZE)
        axes[i, 3].imshow(pred, cmap='hot', vmin=0, vmax=1)
        axes[i, 3].axis('off')
        
        # Col 4: Difference
        axes[i, 4].imshow(gt - pred, cmap='RdBu', vmin=-1, vmax=1)
        axes[i, 4].axis('off')
    
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Epoch {epoch+1} | Batch {batch_idx}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f'e{epoch+1:03d}_b{batch_idx:04d}.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return save_path


# ============== Dataset ==============

class KidneyDistanceDataset(Dataset):
    """
    Dataset for data_combined structure:
    - data_root/images/*.png
    - data_root/masks/*.png
    """
    
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        
        images_dir = self.data_root / "images"
        masks_dir = self.data_root / "masks"
        
        if not images_dir.exists():
            print(f"ERROR: Images directory not found: {images_dir}")
            return
        
        if not masks_dir.exists():
            print(f"ERROR: Masks directory not found: {masks_dir}")
            return
        
        # Find all images with matching masks
        for img_name in os.listdir(images_dir):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = images_dir / img_name
            mask_path = masks_dir / img_name  # Same name
            
            if mask_path.exists():
                self.samples.append((str(img_path), str(mask_path)))
        
        print(f"Dataset: {len(self.samples)} samples")
        
        if len(self.samples) == 0:
            print(f"  Images dir: {images_dir}")
            print(f"  Masks dir: {masks_dir}")
            print(f"  Check that mask filenames match image filenames!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, _depth=0):
        if _depth > 10:
            raise RuntimeError(f"Too many invalid samples near index {idx}")
        
        img_path, mask_path = self.samples[idx]
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"Warning: Could not load {img_path} or {mask_path}")
            return self.__getitem__((idx + 1) % len(self), _depth + 1)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
        
        mask_small = cv2.resize(mask_np, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        distance = compute_distance_transform(mask_small)
        
        if distance.max() < 0.05:
            return self.__getitem__((idx + 1) % len(self), _depth + 1)
        
        distance_tensor = torch.from_numpy(distance).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)
        
        return img, mask_tensor, distance_tensor


# ============== Model ==============

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


# ============== Loss ==============

class DistanceLoss(nn.Module):
    """
    MSE + Peak-weighted MSE + Dice + Boundary-weighted loss.

    Why each term:
      mse_loss        -- base regression signal
      weighted_mse    -- emphasizes tubule centers (high distance values) so
                         peaks in the distance map are sharp and well-localised
      dice_loss       -- ratio-based foreground overlap; small tubules (whose
                         distance values only reach ~0.05-0.15) contribute
                         equally to large ones, fixing the MSE size-bias
      boundary_mse    -- upweights near-zero pixels where touching tubules
                         share a boundary; sharp valleys there are the key
                         signal for instance separation at inference time
    """

    def __init__(self, mse_weight=1.0, peak_weight=1.0, dice_weight=0.5, boundary_weight=2.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.peak_weight = peak_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        # 1. Base MSE
        mse_loss = F.mse_loss(pred, target)

        # 2. Peak-weighted MSE — focus on tubule centres
        weights = 1.0 + target * 4.0
        weighted_mse = (weights * (pred - target) ** 2).mean()

        # 3. Dice loss on soft-binarised distance map.
        #    Threshold at 0.05 so small tubules (max dist ~5 px → normalised ~0.1)
        #    are captured in the foreground mask.
        pred_binary = torch.sigmoid((pred - 0.05) * 30)
        target_binary = (target > 0.05).float()
        intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + 1e-6) / (
            pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3)) + 1e-6
        )
        dice_loss = dice_loss.mean()

        # 4. Boundary-weighted MSE — foreground pixels close to a tubule edge
        #    (distance 0.01–0.15) are the hardest but most critical for splitting.
        near_boundary = ((target > 0.01) & (target < 0.15)).float()
        denom = near_boundary.sum() + 1e-6
        boundary_mse = (near_boundary * (pred - target) ** 2).sum() / denom

        return (
            self.mse_weight * mse_loss
            + self.peak_weight * weighted_mse
            + self.dice_weight * dice_loss
            + self.boundary_weight * boundary_mse
        )


# ============== Feature Extraction ==============

def extract_midnight_features(midnight, images):
    """
    Extracts normalized patch tokens from the OpenMidnight (DINOv2) backbone.
    """
    images_midnight = F.interpolate(images, size=(DINO_SIZE, DINO_SIZE), mode='bilinear', align_corners=False)
    tokens = midnight.forward_features(images_midnight)["x_norm_patchtokens"]
    B = images.shape[0]
    grid = int(tokens.shape[1] ** 0.5)
    return tokens.permute(0, 2, 1).reshape(B, 1536, grid, grid).contiguous()


def extract_sam2_features(sam2_enc, images):
    """
    Extracts multi-scale features from the SAM2 image encoder's feature pyramid.
    """
    output = sam2_enc(images)
    
    for key in ["backbone_fpn", "vision_features", "feature_maps"]:
        if key in output:
            feat = output[key][-1] if isinstance(output[key], list) else output[key]
            break
    else:
        raise KeyError(f"Unknown SAM2 keys: {output.keys()}")
    
    if feat.dim() == 3:
        B = images.shape[0]
        if feat.shape[-1] == 256:
            feat = feat.permute(0, 2, 1)
        H = int(feat.shape[-1] ** 0.5)
        feat = feat.reshape(B, -1, H, H).contiguous()
    
    return feat


# ============== Training ==============

def main():
    """
    Main training loop for the DistanceModel.
    - Loads frozen OpenMidnight and SAM2 backbones.
    - Applies heavy data augmentation for histopathology images.
    - Trains the alignment bridge and distance head.
    """
    print("=" * 60)
    print("Distance Transform Training")
    print("=" * 60)
    
    # Load backbones
    print("\n[1/4] Loading OpenMidnight...")
    dl_loc = hf_hub_download(repo_id="SophontAI/OpenMidnight", filename="teacher_checkpoint_load.pt")
    midnight = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=False)
    cp = torch.load(dl_loc, map_location=DEVICE)
    midnight.pos_embed = nn.Parameter(cp["pos_embed"])
    midnight.load_state_dict(cp)
    midnight = midnight.to(DEVICE).eval()
    for p in midnight.parameters():
        p.requires_grad = False
    print("OpenMidnight loaded")

    print("\n[2/4] Loading SAM2...")
    sam2_enc = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE).image_encoder.eval()
    for p in sam2_enc.parameters():
        p.requires_grad = False
    print("SAM2 loaded")

    print("\n[3/4] Creating model...")
    model = DistanceModel().to(DEVICE)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = DistanceLoss()
    
    print("\n[4/4] Loading data...")
    
    # ============== Heavy Data Augmentation ==============
    transform = A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        
        # Elastic/Grid distortion (good for histology)
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
        ], p=0.3),
        
        # Color augmentation (important for H&E staining variation)
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        ], p=0.5),
        
        # Brightness/Contrast
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Blur/Noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10, 50), p=1.0),
        ], p=0.3),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dataset = KidneyDistanceDataset("./data_combined", transform)
    num_workers = 0 if DEVICE == "mps" else 8
    pin_memory = DEVICE == "cuda"
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    print(f"{len(loader)} batches")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("distance_outputs", exist_ok=True)
    best_loss = float('inf')
    
    print("\n" + "=" * 60)
    print("Training Started")
    print("=" * 60 + "\n")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        t0 = time.time()
        
        for i, (images, masks, targets) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            targets = targets.float().to(DEVICE, non_blocking=True)
            
            # 1. Forward Pass
            with torch.no_grad():
                # Extract backbone features without gradients to save VRAM
                m_feat = extract_midnight_features(midnight, images)
                s_feat = extract_sam2_features(sam2_enc, images)
            
            # Predict distance using AlignmentBridge + DistanceHead
            preds = model(m_feat, s_feat)
            
            # 2. Compute Loss
            if preds.shape != targets.shape:
                targets = F.interpolate(targets, size=preds.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(preds, targets)
            
            # 3. Backward Pass & Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent exploding gradients
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 4. Visualization & Logging
            if i % VISUALIZE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    masks_viz = F.interpolate(masks, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='nearest')
                    save_path = visualize_batch(images, masks_viz, targets, preds, epoch, i)
                print(f"Saved: {save_path.name}")
                model.train()
            
            if i % 10 == 0:
                print(f"E{epoch+1} B{i+1}/{len(loader)} | Loss: {loss.item():.5f}")
        
        avg_loss = epoch_loss / len(loader)
        scheduler.step()
        
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.5f} | Time: {time.time()-t0:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss
            }, "checkpoints/best_distance_model.pt")
            print(f"Best model saved!")
        
        print(f"{'─'*60}\n")
    
    print(f"Training complete! Best loss: {best_loss:.5f}")


if __name__ == "__main__":
    main()