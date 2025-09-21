import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F


def pick_device(user_choice: str = "auto") -> torch.device:
    if user_choice and user_choice.lower() != "auto":
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def uniform_temporal_subsample(video: torch.Tensor, num_samples: int) -> torch.Tensor:
    C, T, H, W = video.shape
    if T == num_samples:
        return video
    if T < 1:
        return video
    idx = torch.linspace(0, T - 1, steps=num_samples).long()
    idx = torch.clamp(idx, 0, T - 1)
    return video[:, idx, :, :]


def short_side_scale(video: torch.Tensor, min_size: int) -> torch.Tensor:
    C, T, H, W = video.shape
    short = min(H, W)
    if short == 0:
        return video
    scale = float(min_size) / float(short)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    frames = []
    for t in range(T):
        frame = video[:, t, :, :].unsqueeze(0)
        frame = F.interpolate(frame, size=(new_h, new_w), mode="bilinear", align_corners=False)
        frames.append(frame.squeeze(0))
    return torch.stack(frames, dim=1)


def center_crop(video: torch.Tensor, size: int) -> torch.Tensor:
    C, T, H, W = video.shape
    th, tw = size, size
    if H < th or W < tw:
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)
        pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
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
    mean_t = torch.tensor(mean, dtype=video.dtype, device=video.device)[:, None, None, None]
    std_t = torch.tensor(std, dtype=video.dtype, device=video.device)[:, None, None, None]
    return (video - mean_t) / std_t


def build_transform():
    def _apply(video: torch.Tensor) -> torch.Tensor:
        video = uniform_temporal_subsample(video, 16)
        video = short_side_scale(video, 256)
        video = center_crop(video, 224)
        video = normalize(video)
        return video
    return _apply


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
    clip = clip.permute(0, 3, 1, 2).float() / 255.0
    clip = clip.permute(1, 0, 2, 3)
    return clip


class VideoClassifier(nn.Module):
    """Enhanced video classifier based on R3D-18."""
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.5):
        super().__init__()
        # Load R3D-18 backbone
        try:
            weights = R3D_18_Weights.KINETICS400_V1
            self.backbone = r3d_18(weights=weights)
        except Exception:
            self.backbone = r3d_18(weights=None)
        
        # Replace the final classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def predict_video(model: nn.Module, video_path: str, device: torch.device, 
                 clip_len: int, samples: int = 4) -> tuple:
    """Predict class probabilities for a video using multiple random clips."""
    transform = build_transform()
    model.eval()
    
    all_probs = []
    for _ in range(samples):
        clip = load_random_clip(video_path, clip_len)
        clip = transform(clip)
        clip = clip.unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = model(clip)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy()[0])
    
    # Average probabilities across all clips
    avg_probs = np.mean(all_probs, axis=0)
    predicted_class = np.argmax(avg_probs)
    confidence = avg_probs[predicted_class]
    
    return predicted_class, confidence, avg_probs


def main():
    parser = argparse.ArgumentParser(description="Classify videos using trained supervised model")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--samples", type=int, default=4, help="Number of random clips to sample")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    parser.add_argument("--show_probs", action="store_true", help="Show class probabilities")
    args = parser.parse_args()
    
    device = pick_device(args.device)
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    class_names = checkpoint.get('class_names', ['drinking', 'not_drinking', 'cheating'])
    clip_len = checkpoint.get('clip_len', 16)
    
    model = VideoClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Make prediction
    try:
        predicted_class, confidence, probs = predict_video(
            model, args.video, device, clip_len, args.samples
        )
        
        print(f"Prediction: {class_names[predicted_class]}")
        print(f"Confidence: {confidence:.4f}")
        
        if args.show_probs:
            print("\\nClass probabilities:")
            for i, (name, prob) in enumerate(zip(class_names, probs)):
                print(f"  {name}: {prob:.4f}")
                
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()