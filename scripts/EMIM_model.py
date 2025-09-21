import os
import csv
import random
import argparse
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F
from sklearn.cluster import KMeans


# -----------------------------
# Video utilities
# -----------------------------
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def list_videos(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(VIDEO_EXTS):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def _device_works_for_3d(device: torch.device) -> bool:
    try:
        x = torch.randn(1, 3, 8, 64, 64, device=device)
        conv = torch.nn.Conv3d(3, 8, kernel_size=3, padding=1).to(device)
        y = conv(x)
        _ = y.mean()
        return True
    except Exception as e:
        print(f"[INFO] Device {device} not suitable for 3D ops: {e}")
        return False


def pick_device(user_choice: str = "auto") -> torch.device:
    if user_choice and user_choice.lower() != "auto":
        return torch.device(user_choice)
    if torch.cuda.is_available() and _device_works_for_3d(torch.device("cuda")):
        return torch.device("cuda")
    # Prefer MPS only if it supports 3D convs; otherwise fallback to CPU
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if _device_works_for_3d(torch.device("mps")):
            return torch.device("mps")
    return torch.device("cpu")


def uniform_temporal_subsample(video: torch.Tensor, num_samples: int) -> torch.Tensor:
    # video: (C, T, H, W)
    C, T, H, W = video.shape
    if T == num_samples:
        return video
    if T < 1:
        return video
    idx = torch.linspace(0, T - 1, steps=num_samples).long()
    idx = torch.clamp(idx, 0, T - 1)
    return video[:, idx, :, :]


def short_side_scale(video: torch.Tensor, min_size: int) -> torch.Tensor:
    # video: (C, T, H, W) -> scale spatial dims preserving aspect ratio
    C, T, H, W = video.shape
    short = min(H, W)
    if short == 0:
        return video
    scale = float(min_size) / float(short)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    frames = []
    for t in range(T):
        frame = video[:, t, :, :].unsqueeze(0)  # (1, C, H, W)
        frame = F.interpolate(frame, size=(new_h, new_w), mode="bilinear", align_corners=False)
        frames.append(frame.squeeze(0))
    return torch.stack(frames, dim=1)


def center_crop(video: torch.Tensor, size: int) -> torch.Tensor:
    # video: (C, T, H, W)
    C, T, H, W = video.shape
    th, tw = size, size
    if H < th or W < tw:
        # pad to at least size
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # (left,right,top,bottom)
        frames = []
        for t in range(T):
            frame = video[:, t, :, :]
            frame = F.pad(frame, pad, mode="constant", value=0)
            frames.append(frame)
        video = torch.stack(frames, dim=1)
        C, T, H, W = video.shape
    i = (H - th) // 2
    j = (W - tw) // 2
    return video[:, :, i : i + th, j : j + tw]


def normalize(video: torch.Tensor, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)) -> torch.Tensor:
    # video: (C, T, H, W)
    mean_t = torch.tensor(mean, dtype=video.dtype, device=video.device)[:, None, None, None]
    std_t = torch.tensor(std, dtype=video.dtype, device=video.device)[:, None, None, None]
    return (video - mean_t) / std_t


def build_transform():
    # returns a callable f(video) -> video
    def _apply(video: torch.Tensor) -> torch.Tensor:
        video = uniform_temporal_subsample(video, 16)
        video = short_side_scale(video, 256)
        video = center_crop(video, 224)
        video = normalize(video)
        return video

    return _apply


def load_random_clip(video_path: str, clip_len: int) -> Optional[torch.Tensor]:
    try:
        # video: (T, H, W, C), uint8
        video, _, _ = read_video(video_path, pts_unit="sec")
    except Exception as e:
        print(f"[WARN] Failed to read {video_path}: {e}")
        return None

    total = video.shape[0]
    if total == 0:
        print(f"[WARN] Empty video {video_path}")
        return None

    if total >= clip_len:
        start = random.randint(0, total - clip_len)
        clip = video[start : start + clip_len]
    else:
        pad = torch.zeros((clip_len - total, *video.shape[1:]), dtype=video.dtype)
        clip = torch.cat([video, pad], dim=0)

    # To (C, T, H, W) in float32 [0,1]
    clip = clip.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
    return clip


def extract_feature_for_clip(model: nn.Module, clip: torch.Tensor, transform, device: torch.device) -> Optional[np.ndarray]:
    try:
        x = transform(clip).unsqueeze(0).to(device)  # (1, C, T, H, W)
        with torch.no_grad():
            feat = model(x)  # (1, D)
        return feat.squeeze(0).detach().cpu().numpy()
    except Exception as e:
        print(f"[WARN] Feature extraction failed: {e}")
        return None


def compute_video_embedding(
    model: nn.Module,
    transform,
    video_path: str,
    clip_len: int,
    clips_per_video: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    feats: List[np.ndarray] = []
    for _ in range(clips_per_video):
        clip = load_random_clip(video_path, clip_len)
        if clip is None:
            continue
        f = extract_feature_for_clip(model, clip, transform, device)
        if f is not None:
            feats.append(f)
    if not feats:
        return None
    return np.mean(np.stack(feats, axis=0), axis=0)


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def infer_folder_label(video_path: str) -> Optional[int]:
    """Infer weak label from parent folder name.
    Returns 0 for drinking, 1 for not_drinking, 2 for cheating, or None if unknown.
    """
    parent = os.path.basename(os.path.dirname(video_path)).lower().replace('-', '_')
    
    # Check for cheating
    if 'cheat' in parent:
        return 2
    
    # Check for drinking categories
    if 'drink' in parent:
        # If any negation present with 'drink', treat as not_drinking
        if any(tok in parent for tok in ['not', 'no', 'non']):
            return 1
        return 0
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Cluster videos into 3 groups (drinking, not_drinking, cheating) using pretrained r3d_18 features + KMeans")
    parser.add_argument("--input_dir", required=True, help="Directory containing videos (mixed or in subfolders)")
    parser.add_argument("--output_dir", default="clusters_output", help="Directory to write clustering results")
    parser.add_argument("--clip_len", type=int, default=16, help="Frames per sampled clip before temporal subsample")
    parser.add_argument("--clips_per_video", type=int, default=8, help="Random clips sampled per video for robust embedding")
    parser.add_argument("--names", nargs=3, default=["drinking", "not_drinking", "cheating"], help="Names to assign to clusters 0..2")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    # Model for feature extraction
    try:
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
    except Exception as e:
        print(f"[WARN] Could not load pretrained weights (offline or API change): {e}\n"
              f"       Falling back to randomly initialized weights.")
        model = r3d_18(weights=None)
    # Replace classifier with identity to get penultimate features
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = build_transform()

    # Collect videos
    videos = list_videos(args.input_dir)
    if not videos:
        print(f"No videos found under: {args.input_dir}")
        return
    print(f"Found {len(videos)} videos. Extracting embeddings...")

    embeddings: List[np.ndarray] = []
    kept_paths: List[str] = []
    for i, vp in enumerate(videos, 1):
        emb = compute_video_embedding(
            model=model,
            transform=transform,
            video_path=vp,
            clip_len=args.clip_len,
            clips_per_video=args.clips_per_video,
            device=device,
        )
        if emb is None:
            print(f"[SKIP] {vp}")
            continue
        embeddings.append(emb)
        kept_paths.append(vp)
        if i % 10 == 0:
            print(f"Processed {i}/{len(videos)} videos...")

    if len(embeddings) < 3:
        print("Not enough valid videos to cluster (need >= 3).")
        return

    X = np.vstack(embeddings)
    print(f"Clustering {X.shape[0]} videos into 3 groups...")
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    # Map KMeans clusters to semantic labels using folder names
    # We evaluate all permutations of (0->drinking/1->not_drinking/2->cheating) and pick the best
    weak = [infer_folder_label(p) for p in kept_paths]
    
    def score_mapping(mapping: List[int]) -> int:
        s = 0
        for cl, wl in zip(labels, weak):
            if wl is None:
                continue
            if mapping[int(cl)] == wl:
                s += 1
        return s
    
    # Try all 6 permutations of 3 classes
    from itertools import permutations
    best_score = -1
    best_mapping = None
    for perm in permutations([0, 1, 2]):
        score = score_mapping(list(perm))
        if score > best_score:
            best_score = score
            best_mapping = list(perm)
    
    # Remap clusters to 0(drinking)/1(not_drinking)/2(cheating)
    remapped = [best_mapping[int(c)] for c in labels]

    # Prepare output dir
    out_root = os.path.abspath(args.output_dir)
    ensure_dirs([out_root])

    # Save cluster centers for inference
    centers_path = os.path.join(out_root, "kmeans.npz")
    # Remap centers according to our best mapping
    remapped_centers = np.zeros_like(kmeans.cluster_centers_)
    for orig_idx, new_idx in enumerate(best_mapping):
        remapped_centers[new_idx] = kmeans.cluster_centers_[orig_idx]
    
    np.savez(centers_path, 
             centers=remapped_centers, 
             names=np.array(args.names))

    # Save CSV
    csv_path = os.path.join(out_root, "clusters.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "cluster", "cluster_name"])  # header
        for vp, c in zip(kept_paths, remapped):
            c = int(c)
            cname = args.names[c]
            writer.writerow([vp, c, cname])

    print(f"Done. Results saved to: {out_root}")
    print(f"- Cluster centers: {centers_path}")
    print(f"- CSV summary: {csv_path}")
    print(f"- Best mapping score: {best_score}/{len([w for w in weak if w is not None])}")


if __name__ == "__main__":
    main()

