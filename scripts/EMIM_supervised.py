import os
import csv
import random
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# -----------------------------
# Video utilities (same as before)
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
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if _device_works_for_3d(torch.device("mps")):
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


def load_random_clip(video_path: str, clip_len: int) -> Optional[torch.Tensor]:
    try:
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

    clip = clip.permute(0, 3, 1, 2).float() / 255.0
    clip = clip.permute(1, 0, 2, 3)
    return clip


def get_label_from_path(video_path: str) -> Optional[int]:
    """Extract label from folder name."""
    parent = os.path.basename(os.path.dirname(video_path)).lower().replace('-', '_')
    
    if 'cheat' in parent:
        return 2
    elif 'drink' in parent:
        if any(tok in parent for tok in ['not', 'no', 'non']):
            return 1  # not_drinking
        return 0  # drinking
    return None


def load_dataset(data_dir: str, clip_len: int) -> Tuple[List[torch.Tensor], List[int], List[str]]:
    """Load all videos and their labels."""
    videos = list_videos(data_dir)
    clips = []
    labels = []
    paths = []
    
    print(f"Loading dataset from {data_dir}...")
    for i, video_path in enumerate(videos):
        label = get_label_from_path(video_path)
        if label is None:
            print(f"[SKIP] Unknown label for {video_path}")
            continue
            
        clip = load_random_clip(video_path, clip_len)
        if clip is None:
            continue
            
        clips.append(clip)
        labels.append(label)
        paths.append(video_path)
        
        if (i + 1) % 5 == 0:
            print(f"Loaded {i + 1}/{len(videos)} videos...")
    
    print(f"Successfully loaded {len(clips)} videos")
    return clips, labels, paths


class VideoClassifier(nn.Module):
    """Enhanced video classifier based on R3D-18."""
    
    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.5):
        super().__init__()
        # Load pretrained R3D-18
        try:
            weights = R3D_18_Weights.KINETICS400_V1
            self.backbone = r3d_18(weights=weights)
        except Exception as e:
            print(f"[WARN] Could not load pretrained weights: {e}")
            self.backbone = r3d_18(weights=None)
        
        # Replace the final classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Add custom classifier with dropout
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


class VideoDataset(torch.utils.data.Dataset):
    """Dataset for video clips."""
    
    def __init__(self, clips: List[torch.Tensor], labels: List[int], transform=None):
        self.clips = clips
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.labels[idx]
        
        if self.transform:
            clip = self.transform(clip)
        
        return clip, label


def train_model(model: nn.Module, train_loader, val_loader, device: torch.device, 
                num_epochs: int = 20, lr: float = 1e-4):
    """Train the video classifier."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    return model


def evaluate_model(model: nn.Module, test_loader, device: torch.device, class_names: List[str]):
    """Evaluate the model and print detailed metrics."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    
    return all_predictions, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train a supervised video classifier for drinking compliance")
    parser.add_argument("--input_dir", required=True, help="Directory containing labeled video folders")
    parser.add_argument("--output_dir", default="supervised_model", help="Directory to save the trained model")
    parser.add_argument("--clip_len", type=int, default=16, help="Frames per clip")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    args = parser.parse_args()
    
    device = pick_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    clips, labels, paths = load_dataset(args.input_dir, args.clip_len)
    
    if len(clips) == 0:
        print("No valid clips found!")
        return
    
    # Print class distribution
    class_names = ["drinking", "not_drinking", "cheating"]
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nClass distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {label_counts.get(i, 0)} videos")
    
    # Split data
    train_clips, val_clips, train_labels, val_labels = train_test_split(
        clips, labels, test_size=args.val_split, random_state=42, stratify=labels
    )
    
    # Create datasets and data loaders
    transform = build_transform()
    train_dataset = VideoDataset(train_clips, train_labels, transform)
    val_dataset = VideoDataset(val_clips, val_labels, transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Create and train model
    model = VideoClassifier(num_classes=3).to(device)
    print(f"\nStarting training with {len(train_clips)} training and {len(val_clips)} validation videos...")
    
    trained_model = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)
    
    # Evaluate on validation set
    print("\nFinal evaluation on validation set:")
    evaluate_model(trained_model, val_loader, device, class_names)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "video_classifier.pth")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'class_names': class_names,
        'clip_len': args.clip_len,
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()