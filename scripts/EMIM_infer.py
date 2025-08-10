import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
    CenterCropVideo,
    NormalizeVideo,
)
from torchvision.transforms import Compose


def pick_device(user_choice: str = "auto") -> torch.device:
    if user_choice and user_choice.lower() != "auto":
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform() -> ApplyTransformToKey:
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(8),
                ShortSideScale(min_size=256),
                CenterCropVideo(224),
                NormalizeVideo(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
            ]
        ),
    )


def load_random_clip(video_path: str, clip_len: int) -> torch.Tensor:
    video, _, _ = read_video(video_path, pts_unit="sec")
    T = video.shape[0]
    if T == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    if T >= clip_len:
        import random as _random
        s = _random.randint(0, T - clip_len)
        clip = video[s : s + clip_len]
    else:
        import torch as _torch
        pad = _torch.zeros((clip_len - T, *video.shape[1:]), dtype=video.dtype)
        clip = torch.cat([video, pad], dim=0)
    clip = clip.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
    return clip


def extract_features(video_path: str, device: torch.device, clip_len: int, samples: int = 4) -> np.ndarray:
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    transform = build_transform()

    feats = []
    for _ in range(samples):
        clip = load_random_clip(video_path, clip_len)
        sample = {"video": clip}
        sample = transform(sample)
        x = sample["video"].unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(x).squeeze(0).detach().cpu().numpy()
        feats.append(f)
    return np.mean(np.stack(feats, axis=0), axis=0)


def main():
    p = argparse.ArgumentParser(description="Classify a single video using saved KMeans centers")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--centers", required=True, help="Path to kmeans.npz produced by EMIM_model.py")
    p.add_argument("--clip_len", type=int, default=16)
    p.add_argument("--samples", type=int, default=4, help="Random clips per video")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    data = np.load(args.centers, allow_pickle=True)
    centers = data["centers"]  # (3, D)
    names = data.get("names", np.array(["drinking", "not_drinking", "cheating"]))

    device = pick_device(args.device)
    feat = extract_features(args.video, device=device, clip_len=args.clip_len, samples=args.samples)
    # Choose nearest center
    dists = np.linalg.norm(centers - feat[None, :], axis=1)
    idx = int(np.argmin(dists))
    print(names[idx])


if __name__ == "__main__":
    main()
