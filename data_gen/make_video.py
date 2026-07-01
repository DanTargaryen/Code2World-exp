#!/usr/bin/env python3
"""Synthesize video from an episode's frames using OpenCV (mp4v).
Usage: python make_video.py <episode_dir> [fps]
"""
import sys, os, glob
import cv2

def main():
    d = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    frames = sorted(glob.glob(os.path.join(d, "frames", "*.png")))
    if not frames:
        print("no frames in", d); sys.exit(1)
    h, w = cv2.imread(frames[0]).shape[:2]
    out = os.path.join(d, "video.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(f))
    vw.release()
    print(f"wrote {out} ({len(frames)} frames, {w}x{h}@{fps:g}fps)")

if __name__ == "__main__":
    main()
