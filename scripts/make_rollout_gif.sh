#!/usr/bin/env bash
# Build a side-by-side GIF from two eval rollout videos (left: success, right: failure).
# Videos come from lerobot-eval (outputs/eval/<job>/videos/<task>/eval_episode_<i>.mp4, 50 fps, 640x480).
# The shorter clip is frozen on its last frame so both sides run for the same time.
#
# Usage:
#   bash scripts/make_rollout_gif.sh <success.mp4> <failure.mp4> <out.gif>
set -eu
success=$1 failure=$2 out=$3
fps=10          # output frame rate (source is 50 fps)
width=320       # width of each panel

frames() { ffprobe -v error -show_entries stream=nb_frames -of csv=p=0 "$1"; }
n_s=$(frames "$success"); n_f=$(frames "$failure")
pad_s=$(( n_f > n_s ? n_f - n_s : 0 ))
pad_f=$(( n_s > n_f ? n_s - n_f : 0 ))

ffmpeg -v error -y -i "$success" -i "$failure" -filter_complex "\
[0:v]tpad=stop_mode=clone:stop=${pad_s},scale=${width}:-1[l];\
[1:v]tpad=stop_mode=clone:stop=${pad_f},scale=${width}:-1[r];\
[l][r]hstack=inputs=2,fps=${fps},split[a][b];\
[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" "$out"
echo "wrote $out ($(du -h "$out" | cut -f1))"
