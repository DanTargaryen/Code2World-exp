"""Render ground-truth Procgen videos for selected variants (same eval seed)."""
import os, sys, subprocess
import numpy as np, cv2

ROOT = "/mnt/pfs/data/huangzehuan/datasets/code2world"
OUT = "/mnt/pfs/users/huangzehuan/projects/linming/workspace/Code2world/model/gt_videos"
os.makedirs(OUT, exist_ok=True)
VARS = ["base", "fast", "highjump"]
EPS = [0, 1, 2]

def load_frames(variant, ep, split="episodes_eval"):
    npz = np.load(os.path.join(ROOT, variant, f"{split}.npz"))
    el = npz["episode_lengths"]
    f0 = int(el[:ep].sum()) + ep
    L = int(el[ep])
    return npz["frames"][f0:f0+L+1], int(npz["seeds"][ep])

def save(frames, path, fps=10):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = frames[0].shape[:2]
    cmd=[ff,"-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(fps),
         "-i","-","-c:v","libx264","-pix_fmt","yuv420p","-crf","16",path]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for f in frames: p.stdin.write(np.ascontiguousarray(f,np.uint8).tobytes())
    p.stdin.close(); p.wait()

def label(img, txt, cell=192):
    img=cv2.resize(img,(cell,cell),interpolation=cv2.INTER_NEAREST)
    bar=np.full((20,cell,3),20,np.uint8)
    cv2.putText(bar,txt,(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(230,230,230),1)
    return np.concatenate([bar,img],0)

for ep in EPS:
    cols_frames={}
    seed=None
    for v in VARS:
        fr,seed=load_frames(v,ep); cols_frames[v]=fr
    L=min(len(cols_frames[v]) for v in VARS)
    out=[]
    for t in range(L):
        cells=[label(cols_frames[v][t], f"{v} (GT)") for v in VARS]
        sep=np.full((cells[0].shape[0],3,3),255,np.uint8)
        row=cells[0]
        for c in cells[1:]: row=np.concatenate([row,sep,c],1)
        out.append(row)
    path=os.path.join(OUT,f"gt_ep{ep}_seed{seed}.mp4")
    save(out,path); print(f"saved {path} ({L} frames, seed={seed})",flush=True)
print("done.")
