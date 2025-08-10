#!/bin/bash

input_dir="/Users/yanpeijie/Library/CloudStorage/OneDrive-TheUniversityofTokyo/Tokyo_University/バイオ物作り/endoscopic_data/original/Not_drinking"
output_dir="$input_dir/frames"

mkdir -p "$output_dir"

for file in "$input_dir"/*.mp4; do

  # Extract frames as JPGs at 10 fps (you can change fps=N if needed)
  ffmpeg -i "$file" -vf fps=10 "$output_dir/frame_%05d.jpg"
done
