# EMIM video clustering and inference

Unsupervised clustering of endoscopic (or general) videos into three groups using pretrained r3d_18 features + KMeans, plus a simple single-video inference script that reuses the learned cluster centers.

Scripts:
- `scripts/EMIM_model.py` — extract features from videos and cluster them into 3 groups; saves a CSV and kmeans centers.
- `scripts/EMIM_infer.py` — classify a single video by nearest KMeans center (no training needed).

## Requirements
- Python 3.12 (Conda environment recommended)
- PyTorch + Torchvision
- scikit-learn
- ffmpeg + av (video decoding for `torchvision.io.read_video`)

Install into your conda environment (example):

```bash
# Activate your environment first
conda activate EMIM

# Core libs
conda install -y pytorch torchvision -c pytorch

# Utilities for clustering and video I/O
conda install -y scikit-learn av ffmpeg -c conda-forge
```

Note: If you already have working torch/torchvision, you may install only `scikit-learn`, `av`, and `ffmpeg`.

## Data layout
The scripts scan videos recursively under the input directory. A structure like the following is fine (names are not required and not used for training; clustering is unsupervised):

```
datasets/
  inside_video_data/
    drinking/            # videos here
    cheating/            # videos here
    not_drinking/        # videos here
```

Any common video formats are supported: `.mp4`, `.mov`, `.avi`, `.mkv`.

## Quick start
Run clustering once to build 3 clusters and save the cluster centers:

```bash
python scripts/EMIM_model.py \
  --input_dir datasets/inside_video_data \
  --output_dir KMeans_Cluster \
  --device auto
```

Outputs in `KMeans_Cluster/`:
- `clusters.csv` — mapping of each video to cluster index and name
- `kmeans.npz` — saved KMeans centers and cluster names (used for inference)

By default, cluster names are `drinking`, `nodrinking`, `cheating`. You can customize:

```bash
python scripts/EMIM_model.py \
  --input_dir datasets/inside_video_data \
  --output_dir KMeans_Cluster \
  --names drinking not_drinking cheating
```

## Single-video inference
Classify a video using the previously saved KMeans centers:

```bash
python scripts/EMIM_infer.py \
  --video datasets/inside_video_data/drinking/1_cheating.mp4 \
  --centers KMeans_Cluster/kmeans.npz \
  --device auto
```

This prints one of the cluster names (e.g., `drinking`, `not_drinking`, `cheating`).

## Useful parameters
- `--clip_len` (default: 16)
  - Number of raw frames per randomly sampled clip, before temporal subsampling.
  - Increase to 32 or 64 to give subsampling more frames to choose from.
- `--clips_per_video` (default: 4)
  - Number of random clips sampled per video; more clips = more robust embedding.
- Temporal subsampling inside the script
  - In `EMIM_model.py` and `EMIM_infer.py`, the transform includes:
    - `uniform_temporal_subsample(video, 8)` — reduces each clip to 8 frames, evenly spaced.
  - To use more frames, change `8` to `16` or `32` in the `build_transform()` function.
- `--device` (auto|cpu|mps|cuda)
  - `auto` tries a working device (prefers GPU/MPS if available and 3D ops work). On Apple Silicon, try `--device mps`.

## How it works
- Feature extractor: torchvision `r3d_18` pretrained on Kinetics (classifier head replaced with Identity to output features).
- For each video:
  - Sample `clips_per_video` random clips of length `clip_len`.
  - Apply transform: temporal subsample → scale short side → center crop (224) → normalize.
  - Extract features for each clip and average them to one embedding per video.
- Cluster embeddings using KMeans (k=3) and save centers for reuse.
- Inference picks the nearest center to the video embedding.

## Troubleshooting
- Video read errors (`read_video`):
  - Ensure `ffmpeg` and `av` are installed (see install commands above).
  - Check video codecs; try re-encoding to H.264 MP4 if needed.
- Torchvision warning about libjpeg:
  - It’s usually harmless for `read_video`. If image ops are needed, install `jpeg/libpng` via conda-forge.
- MPS (Apple Silicon) quirks:
  - If you see device errors, run with `--device cpu` or let `--device auto` choose a working device.
- Not enough videos:
  - KMeans requires ≥ 3 valid embeddings. Ensure at least three readable videos.

## Notes
- Clustering is unsupervised; folder names do not affect cluster assignment, only the `--names` labeling of cluster indices.
- More frames/clips improve quality but increase runtime and memory.

---
If you need this to run as a small GUI or want a CSV-to-plot helper, open an issue or ask and we can add it.
